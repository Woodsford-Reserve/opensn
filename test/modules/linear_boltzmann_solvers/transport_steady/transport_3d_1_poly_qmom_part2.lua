-- 3D Transport test with Vacuum BCs and a material source reading source moments
-- SDM: PWLD
-- Test: Max-value=1.01701e-04 and 9.14681e-06
num_procs = 4





--############################################### Check num_procs
if (check_num_procs==nil and number_of_processes ~= num_procs) then
  chiLog(LOG_0ERROR,"Incorrect amount of processors. " ..
    "Expected "..tostring(num_procs)..
    ". Pass check_num_procs=false to override if possible.")
  os.exit(false)
end

--############################################### Setup mesh
meshgen1 = mesh.ExtruderMeshGenerator.Create
({
  inputs =
  {
    mesh.FromFileMeshGenerator.Create
    ({
      filename = "../../../../resources/TestMeshes/SquareMesh2x2Quads.obj"
    }),
  },
  layers = {{z=0.4,n=2},{z=0.8,n=2},{z=1.2,n=2},{z=1.6,n=2}}, -- layers
  partitioner = chi.KBAGraphPartitioner.Create
  ({
    nx = 2, ny=2,
    xcuts = {0.0}, ycuts = {0.0}
  })
})
mesh.MeshGenerator.Execute(meshgen1)

--############################################### Set Material IDs
vol0 = mesh.RPPLogicalVolume.Create({infx=true, infy=true, infz=true})
VolumeMesherSetProperty(MATID_FROMLOGICAL,vol0,0)

vol1 = mesh.RPPLogicalVolume.Create
({ xmin=-0.5/8,xmax=0.5/8,ymin=-0.5/8,ymax=0.5/8, infz=true })
VolumeMesherSetProperty(MATID_FROMLOGICAL,vol1,1)

--############################################### Add materials
materials = {}
materials[1] = chiPhysicsAddMaterial("Test Material");
materials[2] = chiPhysicsAddMaterial("Test Material2");

chiPhysicsMaterialAddProperty(materials[1],TRANSPORT_XSECTIONS)
chiPhysicsMaterialAddProperty(materials[2],TRANSPORT_XSECTIONS)

chiPhysicsMaterialAddProperty(materials[1],ISOTROPIC_MG_SOURCE)
chiPhysicsMaterialAddProperty(materials[2],ISOTROPIC_MG_SOURCE)


num_groups = 21
chiPhysicsMaterialSetProperty(materials[1],TRANSPORT_XSECTIONS,
  CHI_XSFILE,"xs_graphite_pure.cxs")
chiPhysicsMaterialSetProperty(materials[2],TRANSPORT_XSECTIONS,
  CHI_XSFILE,"xs_graphite_pure.cxs")

src={}
for g=1,num_groups do
  src[g] = 0.0
end

chiPhysicsMaterialSetProperty(materials[1],ISOTROPIC_MG_SOURCE,FROM_ARRAY,src)
src[1] = 1.0
chiPhysicsMaterialSetProperty(materials[2],ISOTROPIC_MG_SOURCE,FROM_ARRAY,src)

--############################################### Setup Physics
pquad0 = chiCreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV,2, 2)

lbs_block =
{
  num_groups = num_groups,
  groupsets =
  {
    {
      groups_from_to = {0, 20},
      angular_quadrature_handle = pquad0,
      --angle_aggregation_type = "single",
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
  scattering_order = 1,
  use_source_moments = true,
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
lbs.SetOptions(phys1, lbs_options)

--############################################### Initialize and Execute Solver
ss_solver = lbs.SteadyStateSolver.Create({lbs_solver_handle = phys1})
SolverInitialize(ss_solver)
chiLBSReadSourceMoments(phys1,"Qmoms")

chiSolverExecute(ss_solver)

--############################################### Get field functions
fflist,count = chiLBSGetScalarFieldFunctionList(phys1)

--############################################### Slice plot
--slices = {}
--for k=1,count do
--    slices[k] = chiFFInterpolationCreate(SLICE)
--    chiFFInterpolationSetProperty(slices[k],SLICE_POINT,0.0,0.0,0.8001)
--    chiFFInterpolationSetProperty(slices[k],ADD_FIELDFUNCTION,fflist[k])
--    --chiFFInterpolationSetProperty(slices[k],SLICE_TANGENT,0.393,1.0-0.393,0)
--    --chiFFInterpolationSetProperty(slices[k],SLICE_NORMAL,-(1.0-0.393),-0.393,0.0)
--    --chiFFInterpolationSetProperty(slices[k],SLICE_BINORM,0.0,0.0,1.0)
--    chiFFInterpolationInitialize(slices[k])
--    chiFFInterpolationExecute(slices[k])
--    chiFFInterpolationExportPython(slices[k])
--end

--############################################### Volume integrations
ffi1 = chiFFInterpolationCreate(VOLUME)
curffi = ffi1
chiFFInterpolationSetProperty(curffi,OPERATION,OP_MAX)
chiFFInterpolationSetProperty(curffi,LOGICAL_VOLUME,vol0)
chiFFInterpolationSetProperty(curffi,ADD_FIELDFUNCTION,fflist[1])

chiFFInterpolationInitialize(curffi)
chiFFInterpolationExecute(curffi)
maxval = chiFFInterpolationGetValue(curffi)

chiLog(LOG_0,string.format("Max-value1=%.5e", maxval))

ffi1 = chiFFInterpolationCreate(VOLUME)
curffi = ffi1
chiFFInterpolationSetProperty(curffi,OPERATION,OP_MAX)
chiFFInterpolationSetProperty(curffi,LOGICAL_VOLUME,vol0)
chiFFInterpolationSetProperty(curffi,ADD_FIELDFUNCTION,fflist[20])

chiFFInterpolationInitialize(curffi)
chiFFInterpolationExecute(curffi)
maxval = chiFFInterpolationGetValue(curffi)

chiLog(LOG_0,string.format("Max-value2=%.5e", maxval))

--############################################### Exports
if (master_export == nil) then
  chiExportMultiFieldFunctionToVTK(fflist,"ZPhi3DColl")
end

--############################################### Plots
if (location_id == 0 and master_export == nil) then

  --os.execute("python ZPFFI00.py")
  ----os.execute("python ZPFFI11.py")
  --local handle = io.popen("python ZPFFI00.py")
  print("Execution completed")
end
chiMPIBarrier()
if (location_id == 0) then
  os.execute("rm Qmoms*")
end
