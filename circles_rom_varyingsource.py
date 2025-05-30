#!/usr/bin/env python3
import argparse, math
from dolfin import *
from mshr import Circle, generate_mesh
import matplotlib.pyplot as plt

# ------------ CLI ----------------------------------------------------
p = argparse.ArgumentParser(description="Eigen-ROM on 2-circle annulus")
p.add_argument("--modes", "-m", type=int, default=5,
               help="Number of eigen-modes in the ROM")
args = p.parse_args()
NROM = args.modes
print(f"\nBuilding ROM with {NROM} mode(s)\n")

# ------------ 1) Geometry & mesh -----------------------------------
R_outer, hot_r, cold_r = 12.0, 1.0, 0.5
outer = Circle(Point(0,0), R_outer)
hot   = Circle(Point(0,0), hot_r)
cold  = Circle(Point(2.5,0), cold_r)
domain = outer - hot - cold
mesh   = generate_mesh(domain, 50)

# ------------ 2) Boundary tagging ----------------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
tol = 1e-1
class Hot(SubDomain):
    def inside(self, x, on_b):
        return on_b and near(x[0]**2 + x[1]**2, hot_r**2, tol)
class Cold(SubDomain):
    def inside(self, x, on_b):
        return on_b and near((x[0]-2.5)**2 + x[1]**2, cold_r**2, tol)

Hot().mark(boundaries, 2)
Cold().mark(boundaries, 3)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ------------ 3) Function space & BCs ------------------------------
V = FunctionSpace(mesh, "CG", 1)
print("DOFs =", V.dim())

u, v   = TrialFunction(V), TestFunction(V)

hot_bc  = DirichletBC(V, Constant(50.0), boundaries, 2)
cold_bc = DirichletBC(V, Constant( 0.0), boundaries, 3)
bcs     = [hot_bc, cold_bc]

# ------------ 4) Forms & steady check ------------------------------
a_form = dot(grad(u), grad(v))*dx
m_form = u*v*dx
L0     = Constant(0.0)*v*dx

phi_h = Function(V)
A0 = assemble(a_form)
b0 = assemble(L0)
for bc in bcs:
    bc.apply(A0, b0)
solve(A0, phi_h.vector(), b0)

# ------------ 5) lifting field w -----------------------------------
w = Function(V)
solve(a_form == Constant(0.0)*v*dx, w, bcs=bcs)

# ------------ 6) full FEM with variable sink ----------------------
# define f(x,y) = x^3 * y
f_sink = Expression("1e-2*pow(x[0],3)*x[1]", degree=4)
F_full = assemble(f_sink*v*dx)

A_raw  = PETScMatrix(); assemble(a_form, tensor=A_raw)
b_full = F_full.copy()
for bc in bcs:
    bc.apply(A_raw, b_full)

u_fem = Function(V)
solve(A_raw, u_fem.vector(), b_full)

# ------------ 7) build residual RHS b_res = F_full - A_raw*w -------
M     = PETScMatrix(); assemble(m_form, tensor=M)
b_res = F_full.copy()
b_res.axpy(-1.0, A_raw*w.vector())   # b_res = F_full - A_raw*w

# ------------ 8) impose homogeneous BCs ----------------------------
for bc in bcs:
    bc.apply(A_raw)
    bc.apply(M)
    bc.apply(b_res)

# ------------ 9) solve eigenproblem A_raw φ = λ M φ ---------------
nev = max(2*NROM, NROM)
eigs = SLEPcEigenSolver(A_raw, M)
eigs.parameters["spectrum"]   = "smallest real"
eigs.parameters["tolerance"]   = 1e-10
eigs.solve(nev)
nconv = eigs.get_number_converged()
print(f"Converged eigenpairs: {nconv}")

# ------------ 10) ROM assembly by Galerkin projection --------------
u_r_vec = PETScVector(M.mpi_comm(), M.size(0)); u_r_vec.zero()
print("\n mode   lambda         coeff")
print(" -------------------------------")
for i in range(min(NROM, nconv)):
    lam, _, raw_phi, _ = eigs.get_eigenpair(i)
    phi = raw_phi.copy()
    phi /= math.sqrt(phi.inner(M*phi))
    coeff = phi.inner(b_res) / lam
    u_r_vec.axpy(coeff, phi)
    print(f"{i:3d}  {lam:12.4e}  {coeff:12.4e}")

# ------------ 11) reconstruct + error -----------------------------
u_rom = Function(V)
u_rom.vector()[:] = w.vector() + u_r_vec

rel_err = errornorm(u_fem, u_rom, 'L2')/norm(u_fem, 'L2')
print(f"\nRelative L2 error with {min(NROM,nconv)} modes: {rel_err:.3e}\n")

# ------------ 12) plots --------------------------------------------
plt.figure(); m1 = plot(u_fem, title="Full FEM"); plt.colorbar(m1)
plt.figure(); m2 = plot(u_rom, title=f"{min(NROM,nconv)}-mode ROM"); plt.colorbar(m2)
plt.show()

