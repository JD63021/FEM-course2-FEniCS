#import the packages # define geometry #build geometry & mesh & mark boundaries 
# define functionspaces& assembly & solve # postprocessing
from dolfin import*
from mshr import Circle, generate_mesh
import math
import numpy as np
import matplotlib.pyplot as plt

# geometry
R_outer = 50
hot_r = 1
cold_r = 0.5

outer = Circle(Point(0,0),R_outer)
hot = Circle(Point(0,0),hot_r)
cold = Circle(Point(2.5,0),cold_r)
domain = outer - hot - cold
mesh = generate_mesh(domain,300)

#mark boundaries
boundaries = MeshFunction("size_t",mesh,mesh.topology().dim() - 1, 0)
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
ds = Measure("ds", domain = mesh, subdomain_data = boundaries)

# Function
V= FunctionSpace(mesh,"Lagrange",2)
phi,v = TrialFunction(V), TestFunction(V)

#weak form
Bf = dot(grad(phi),grad(v))*dx  
Lf = Constant(0)*v*dx

#assemble & impose BCs & solve
B = assemble(Bf)
L = assemble(Lf)
hot_bc = DirichletBC(V,Constant(50),boundaries,2)
cold_bc = DirichletBC(V,Constant(0),boundaries,3)

for bc in (hot_bc, cold_bc):
	bc.apply(B,L)
	
phi_h = Function(V)
solve(B,phi_h.vector(),L)

# post process
#flux  via boudnary integral
normal = FacetNormal(mesh)
flux_int = assemble(-dot(grad(phi_h),normal)*ds(2))
print(f"Flux via boundary integral:{flux_int:.6f}")

#Flux via reaction forces
B0 = assemble(Bf)
L0 = assemble(Lf)
reac = B0*phi_h.vector() - L0
flux_reac = sum(reac[i] for i in hot_bc.get_boundary_values())
print(f"Flux via reaction forces:{flux_reac:.6f}")

# plots
plt.figure()
plot(phi_h)
plot(mesh, linewidth=0.3)


plt.show()













