// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "lua/framework/lua.h"
#include "lua/framework/console/console.h"
#include "lua/framework/math/quadratures/sldfesq/sldfe.h"
#include "framework/math/quadratures/angular/sldfe_sq_quadrature.h"
#include "framework/runtime.h"

using namespace opensn;

namespace opensnlua
{

RegisterLuaFunctionInNamespace(CreateSLDFESQAngularQuadrature,
                               aquad,
                               CreateSLDFESQAngularQuadrature);

int
CreateSLDFESQAngularQuadrature(lua_State* L)
{
  LuaCheckArgs<int>(L, "aquad.CreateSLDFESQAngularQuadrature");

  auto init_refinement_level = LuaArg<int>(L, 1);

  auto sldfesq = std::make_shared<SimplifiedLDFESQ::Quadrature>();
  sldfesq->GenerateInitialRefinement(init_refinement_level);

  opensn::angular_quadrature_stack.push_back(sldfesq);
  const size_t index = opensn::angular_quadrature_stack.size() - 1;
  return LuaReturn(L, index);
}

} // namespace opensnlua
