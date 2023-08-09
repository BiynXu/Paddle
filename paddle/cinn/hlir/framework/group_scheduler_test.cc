// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/hlir/framework/group_scheduler.h"

#include <gtest/gtest.h>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/decomposer/test_helper.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"

DECLARE_bool(cinn_new_group_scheduler);

namespace cinn {
namespace hlir {
namespace framework {

using frontend::NetBuilder;
using frontend::RunDecomposer;

void Compile(NetBuilder* net_builder) {
  auto program = net_builder->Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
          "inferdtype");
  auto& shape_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
          "infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_group : graph->fusion_groups) {
    std::vector<ir::LoweredFunc> lowered_funcs =
        op_lowerer.Lower(fusion_group,
                         /* apply_op_schedule = */ true,
                         /* apply_group_schedule = */ false);
    CHECK_EQ(lowered_funcs.size(), 1);
    VLOG(1) << "without group schedule, lowered_func: "
            << lowered_funcs.front();
    // ir::Expr func_body = lowered_funcs.front()->body;
    // ir::ModuleExpr mod_expr({func_body});
    // ir::IRSchedule ir_sch(mod_expr);
    // GroupScheduler group_scheduler(&ir_sch, fusion_group, target);
    // group_scheduler();
    FLAGS_cinn_new_group_scheduler = true;
    lowered_funcs = op_lowerer.Lower(fusion_group,
                                     /* apply_op_schedule = */ true,
                                     /* apply_group_schedule = */ true);
    CHECK_EQ(lowered_funcs.size(), 1);
    VLOG(1) << "after group schedule, lowered_func: " << lowered_funcs.front();
  }
}

void CheckAccuracy(NetBuilder* net_builder,
                   const std::vector<std::string>& input_names) {
  FLAGS_cinn_new_group_scheduler = true;
  auto program = net_builder->Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  for (size_t i = 0; i < input_names.size(); ++i) {
    scope->Var<hlir::framework::Tensor>(input_names[i]);
    auto tensor = scope->GetTensor(input_names[i]);

    std::vector<float> vec;
    frontend::InitRandomVector<float>(
        &vec, tensor->shape().numel(), 0.0f, 1.0f);
    frontend::CopyFromVector<float>(vec, tensor, target);
  }

  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(GROUP_SCHEDULER, last_reduce_only_1) {
  NetBuilder net_builder("last_reduce_only_1");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {128, 64, 32}, "A");
    auto B = net_builder.ReduceSum(A, {2});
  };

  CreateModel();
  Compile(&net_builder);
}

// TEST(GROUP_SCHEDULER, elementwise_1) {
//   int h = 128, w = 128;
//   NetBuilder net_builder("elementwise_1");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
//     auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
//     auto C = net_builder.Add(A, B);
//     auto D = net_builder.Add(B, C);
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, elementwise_2) {
//   int h = 128, w = 128;
//   NetBuilder net_builder("elementwise_2");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
//     auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
//     auto C = net_builder.Add(A, B);
//     auto D = net_builder.Cast(C, "float16");
//     auto E = net_builder.Cast(C, "float16");
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, elementwise_3) {
//   int h = 128, w = 128;
//   NetBuilder net_builder("elementwise_3");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
//     auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
//     auto C = net_builder.Add(A, B);
//     auto D = net_builder.Cast(C, "float16");
//     auto E = net_builder.Cast(C, "float16");
//     auto F = net_builder.Cast(D, "float32");
//     auto G = net_builder.Cast(E, "float32");
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, elementwise_4) {
//   int h = 128, w = 128;
//   NetBuilder net_builder("elementwise_4");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
//     auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
//     auto C = net_builder.Add(A, B);
//     auto D = net_builder.Cast(C, "float16");
//     auto E = net_builder.Cast(C, "float16");
//     auto F = net_builder.Add(D, E);
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, elementwise_broadcast) {
//   NetBuilder net_builder("elementwise_broadcast");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {128}, "A");
//     auto B = net_builder.CreateInput(Float(32), {128}, "B");
//     auto C = net_builder.Add(A, B);
//     auto D = net_builder.BroadcastTo(C, {128, 128});
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, elementwise_double_broadcast) {
//   NetBuilder net_builder("elementwise_double_broadcast");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {128}, "A");
//     auto B = net_builder.CreateInput(Float(32), {128}, "B");
//     auto C = net_builder.Add(A, B);
//     auto D = net_builder.BroadcastTo(C, {128, 128});
//     auto E = net_builder.BroadcastTo(C, {128, 128});
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, non_last_reduce_elementwise) {
//   int h = 128, w = 128;
//   NetBuilder net_builder("non_last_reduce_elementwise");
//   std::vector<std::string> input_names = {"A", "B"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
//     auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
//     auto C = net_builder.Add(A, B);
//     auto E = net_builder.ReduceSum(C, {0});
//     auto F = net_builder.Cast(E, "float16");
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

// TEST(GROUP_SCHEDULER, last_reduce_elementwise) {
//   NetBuilder net_builder("last_reduce_elementwise");
//   std::vector<std::string> input_names = {"A", "C"};
//   // create model
//   auto CreateModel = [&]() {
//     auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
//     auto B = net_builder.ReduceSum(A, {1});
//     auto C = net_builder.CreateInput(Float(32), {128}, "C");
//     auto D = net_builder.Add(B, C);
//   };

//   CreateModel();
//   Compile(&net_builder);
//   CreateModel();
//   CheckAccuracy(&net_builder, input_names);
// }

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
