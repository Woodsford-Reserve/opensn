-- 3D LinearBSolver test of a block of graphite with an air cavity. DSA and TG
-- SDM: PWLD
-- Test: WGS groups [0-62] Iteration    54 Residual 7.92062e-07 CONVERGED
-- and   WGS groups [63-167] Iteration    63 Residual 8.1975e-07 CONVERGED
num_procs = 4

-- Check num_procs
if check_num_procs == nil and number_of_processes ~= num_procs then
  log.Log(
    LOG_0ERROR,
    "Incorrect amount of processors. "
      .. "Expected "
      .. tostring(num_procs)
      .. ". Pass check_num_procs=false to override if possible."
  )
  os.exit(false)
end

-- Setup mesh
nodes = {}
N = 20
L = 100.0
--N=10
--L=200e6
xmin = -L / 2
--xmin = 0.0
dx = L / N
for i = 1, (N + 1) do
  k = i - 1
  nodes[i] = xmin + k * dx
end
znodes = { 0.0, 10.0, 20.0, 30.0, 40.0 }

meshgen1 = mesh.OrthogonalMeshGenerator.Create({ node_sets = { nodes, nodes, znodes } })
meshgen1:Execute()

-- Set Material IDs
mesh.SetUniformMaterialID(0)

vol1 = logvol.RPPLogicalVolume.Create({
  xmin = -10.0,
  xmax = 10.0,
  ymin = -10.0,
  ymax = 10.0,
  infz = true,
})
mesh.SetMaterialIDFromLogicalVolume(vol1, 1, true)

-- Add materials
materials = {}
materials[1] = mat.AddMaterial("Test Material")
materials[2] = mat.AddMaterial("Test Material2")

num_groups = 168
xs_graphite = xs.LoadFromOpenSn("xs_graphite_pure.xs")
materials[1]:SetTransportXSections(xs_graphite)
xs_air = xs.LoadFromOpenSn("xs_air50RH.xs")
materials[2]:SetTransportXSections(xs_air)

src = {}
for g = 1, num_groups do
  src[g] = 0.0
end
src[1] = 1.0
mg_src0 = xs.IsotropicMultiGroupSource.FromArray(src)
materials[1]:SetIsotropicMGSource(mg_src0)
src[1] = 0.0
mg_src1 = xs.IsotropicMultiGroupSource.FromArray(src)
materials[2]:SetIsotropicMGSource(mg_src1)

-- Setup Physics
pquad0 = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV, 2, 2)

lbs_block = {
  num_groups = num_groups,
  groupsets = {
    {
      groups_from_to = { 0, 62 },
      angular_quadrature = pquad0,
      angle_aggregation_num_subsets = 1,
      groupset_num_subsets = 1,
      inner_linear_method = "petsc_gmres",
      l_abs_tol = 1.0e-6,
      l_max_its = 1000,
      gmres_restart_interval = 30,
      apply_wgdsa = true,
      wgdsa_l_abs_tol = 1.0e-2,
    },
    {
      groups_from_to = { 63, num_groups - 1 },
      angular_quadrature = pquad0,
      angle_aggregation_num_subsets = 1,
      groupset_num_subsets = 1,
      inner_linear_method = "petsc_gmres",
      l_abs_tol = 1.0e-6,
      l_max_its = 1000,
      gmres_restart_interval = 30,
      apply_wgdsa = true,
      apply_tgdsa = true,
      wgdsa_l_abs_tol = 1.0e-2,
    },
  },
}

lbs_options = {
  scattering_order = 1,
  max_ags_iterations = 1,
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
phys1:SetOptions(lbs_options)

-- Initialize and Execute Solver
ss_solver = lbs.SteadyStateSolver.Create({ lbs_solver = phys1 })

ss_solver:Initialize()
ss_solver:Execute()

-- Get field functions
fflist = lbs.GetScalarFieldFunctionList(phys1)

-- Exports
if master_export == nil then
  fieldfunc.ExportToVTKMulti(fflist, "ZPhi")
end

-- Plots
