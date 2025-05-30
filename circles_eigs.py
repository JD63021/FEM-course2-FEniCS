#import the packages# define the geomtry # build the geomtry & mesh & tag boundaries # define #function spaces # assemble apply BCs & solve # postprocess to compute fluxes

from dolfin import*
from mshr import Circle, generate_mesh
import math
import numpy as np
import matplotlib.pyplot as plt

# geometry
R_outer = 12
hot_r = 1
cold_r = 0.5

outer = Circle(Point(0,0),R_outer)
hot = Circle(Point(0,0),hot_r)
cold = Circle(Point(2.5,0),cold_r)
domain = outer - hot - cold
mesh = generate_mesh(domain,50)

# mark boundaries
boundaries = MeshFunction("size_t",mesh, mesh.topology().dim() - 1,0)
tol = 0.1
class Hot(SubDomain):
	def inside(self, x, on_b):
		return on_b and near(x[0]**2 + x[1]**2,hot_r**2,tol)
class Cold(SubDomain):
	def inside(self, x, on_b):
		return on_b and near((x[0]-2.5)**2 + x[1]**2,cold_r**2,tol)		
class Outer(SubDomain):
	def inside(self, x, on_b):
		return on_b and near(x[0]**2 + x[1]**2,R_outer**2,tol)
		
Outer().mark(boundaries,1)
Hot().mark(boundaries,2)
Cold().mark(boundaries,3)
 
ds = Measure("ds", domain = mesh, subdomain_data=boundaries)
 
 #Function spaces
 
V = FunctionSpace(mesh,"Lagrange", 1)
phi, v = TrialFunction(V), TestFunction(V)
 
Bf = dot(grad(phi),grad(v))*dx 
Lf =Constant(0)*v*dx
Mf = phi*v*dx 

B = assemble(Bf)
L = assemble(Lf)

hot_bc = DirichletBC(V,Constant(50),boundaries,2)
cold_bc = DirichletBC(V,Constant(0),boundaries,3)

for bc in (hot_bc,cold_bc):
	bc.apply(B,L)
	
phi_h = Function(V)
solve(B,phi_h.vector(),L)

#postprocess
#boundary integral approach
normal = FacetNormal(mesh)
flux_int = assemble(-dot(grad(phi_h),normal)*ds(2))
print(f"Flux boundary integral:{flux_int:.6f}")
#reaction forces approach
B0 = assemble(Bf)
L0 = assemble(Lf)
reac = B0*phi_h.vector() - L0
flux_reac = sum(reac[i] for i in hot_bc.get_boundary_values())
print(f"Flux reaction forces:{flux_reac:.6f}")

#plots
plt.figure()
plot(mesh,linewidth=0.3)
plt.figure()
plot(phi_h)
plt.show()


# ------------------------------------------------------------------
# Eigenvalue problem  A u = λ M u -----------------------------------
# ------------------------------------------------------------------
# Assemble matrices
A = PETScMatrix(); assemble(Bf, tensor=A)
M = PETScMatrix(); assemble(Mf, tensor=M)

# Impose Dirichlet BCs on eigenproblem
for bc in (hot_bc, cold_bc):
    bc.apply(A)
    bc.apply(M)

# Solve for smallest 5 eigenpairs
nev = 5
eigensolver = SLEPcEigenSolver(A, M)
eigensolver.parameters["spectrum"]  = "smallest real"
eigensolver.parameters["tolerance"] = 1e-9
eigensolver.solve(nev)

nconv = eigensolver.get_number_converged()
print(f"\nConverged eigenpairs: {nconv}\n")

# ------------------------------------------------------------------
# Plots & diagnostics ---------------------------------------------
# ------------------------------------------------------------------
print(f"Steady ϕ  range : {phi_h.vector().min():.3f} – {phi_h.vector().max():.3f}")
for i in range(min(nev, nconv)):
    λ, _, rx, _ = eigensolver.get_eigenpair(i)
    print(f"Mode {i}:  λ = {λ:.6e}")
    ui = Function(V)
    ui.vector()[:] = rx
    plt.figure()
    p = plot(ui, title=f"Eigenmode {i} (λ={λ:.2e})")
    plt.colorbar(p)
    plt.xlabel("x"); plt.ylabel("y")

# Mesh & steady solution
plt.figure()
plot(mesh, linewidth=0.3)
plt.title("Mesh")

plt.figure()
p0 = plot(phi_h, title="Steady φ")
plt.colorbar(p0)
plt.xlabel("x"); plt.ylabel("y")

plt.show()






































 
 
 
