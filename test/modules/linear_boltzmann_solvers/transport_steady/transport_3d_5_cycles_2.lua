-- 3D Transport test with Vacuum BCs.
-- SDM: PWLD
-- Test: Max-value1=6.55387e+00
--       Max-value2=1.02940e+00

num_procs = 4





--############################################### Check num_procs
if (check_num_procs==nil and number_of_processes ~= num_procs) then
  log.Log(LOG_0ERROR,"Incorrect amount of processors. " ..
    "Expected "..tostring(num_procs)..
    ". Pass check_num_procs=false to override if possible.")
  os.exit(false)
end

--############################################### Setup mesh
meshgen1 = mesh.MeshGenerator.Create
({
  inputs =
  {
    mesh.FromFileMeshGenerator.Create
    ({
      filename = "../../../../resources/TestMeshes/Sphere.case"
    }),
  },
  partitioner = KBAGraphPartitioner.Create
  ({
    nx = 2, ny=2, nz=1,
    xcuts = {0.0}, ycuts = {0.0}
  })
})
mesh.MeshGenerator.Execute(meshgen1)

--############################################### Add materials
materials = {}
materials[1] = mat.AddMaterial("Test Material");
materials[2] = mat.AddMaterial("Test Material2");

mat.AddProperty(materials[1], TRANSPORT_XSECTIONS)
mat.AddProperty(materials[2], TRANSPORT_XSECTIONS)

mat.AddProperty(materials[1], ISOTROPIC_MG_SOURCE)
mat.AddProperty(materials[2], ISOTROPIC_MG_SOURCE)

num_groups = 5
mat.SetProperty(materials[1], TRANSPORT_XSECTIONS, OPENSN_XSFILE, "xs_graphite_pure.xs")
mat.SetProperty(materials[2], TRANSPORT_XSECTIONS, OPENSN_XSFILE, "xs_graphite_pure.xs")

src={}
for g=1,num_groups do
  src[g] = 0.0
end

mat.SetProperty(materials[2], ISOTROPIC_MG_SOURCE, FROM_ARRAY, src)
src[1]=1.0
mat.SetProperty(materials[1], ISOTROPIC_MG_SOURCE, FROM_ARRAY, src)

--############################################### Setup Physics
pquad0 = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV, 2, 2)

lbs_block =
{
  num_groups = num_groups,
  groupsets =
  {
    {
      groups_from_to = {0, num_groups-1},
      angular_quadrature_handle = pquad0,
      angle_aggregation_type = "single",
      angle_aggregation_num_subsets = 1,
      groupset_num_subsets = 1,
      inner_linear_method = "gmres",
      l_abs_tol = 1.0e-6,
      l_max_its = 300,
      gmres_restart_interval = 100,
    },
  }
}

lbs_options =
{
  scattering_order = 0,
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
lbs.SetOptions(phys1, lbs_options)

--############################################### Initialize and Execute Solver
ss_solver = lbs.SteadyStateSolver.Create({lbs_solver_handle = phys1})

solver.Initialize(ss_solver)
solver.Execute(ss_solver)

--############################################### Get field functions
fflist,count = lbs.GetScalarFieldFunctionList(phys1)

--############################################### Slice plot
--slices = {}
--for k=1,count do
--    slices[k] = fieldfunc.FFInterpolationCreate(SLICE)
--    fieldfunc.SetProperty(slices[k],SLICE_POINT,0.0,0.0,0.8001)
--    fieldfunc.SetProperty(slices[k],ADD_FIELDFUNCTION,fflist[k])
--    --fieldfunc.SetProperty(slices[k],SLICE_TANGENT,0.393,1.0-0.393,0)
--    --fieldfunc.SetProperty(slices[k],SLICE_NORMAL,-(1.0-0.393),-0.393,0.0)
--    --fieldfunc.SetProperty(slices[k],SLICE_BINORM,0.0,0.0,1.0)
--    fieldfunc.Initialize(slices[k])
--    fieldfunc.Execute(slices[k])
--    fieldfunc.ExportPython(slices[k])
--end

--############################################### Volume integrations
vol0 = logvol.RPPLogicalVolume.Create({infx=true, infy=true, infz=true})
ffi1 = fieldfunc.FFInterpolationCreate(VOLUME)
curffi = ffi1
fieldfunc.SetProperty(curffi,OPERATION,OP_MAX)
fieldfunc.SetProperty(curffi,LOGICAL_VOLUME,vol0)
fieldfunc.SetProperty(curffi,ADD_FIELDFUNCTION,fflist[1])

fieldfunc.Initialize(curffi)
fieldfunc.Execute(curffi)
maxval = fieldfunc.GetValue(curffi)

log.Log(LOG_0,string.format("Max-value1=%.5e", maxval))

ffi1 = fieldfunc.FFInterpolationCreate(VOLUME)
curffi = ffi1
fieldfunc.SetProperty(curffi,OPERATION,OP_MAX)
fieldfunc.SetProperty(curffi,LOGICAL_VOLUME,vol0)
fieldfunc.SetProperty(curffi,ADD_FIELDFUNCTION,fflist[2])

fieldfunc.Initialize(curffi)
fieldfunc.Execute(curffi)
maxval = fieldfunc.GetValue(curffi)

log.Log(LOG_0,string.format("Max-value2=%.5e", maxval))

--############################################### Exports
if (master_export == nil) then
  fieldfunc.ExportToVTKMulti(fflist,"ZPhi3D")
  fieldfunc.ExportToVTK(fflist[1],"ZPhi3D_g0")
end

--############################################### Plots
if (location_id == 0 and master_export == nil) then
  print("Execution completed")
end
