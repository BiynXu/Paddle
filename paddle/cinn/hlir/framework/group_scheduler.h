// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#pragma once
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/ir/ir_schedule.h"
#include "paddle/cinn/ir/schedule_block_graph.h"

namespace cinn {
namespace hlir {
namespace framework {

class GroupScheduler {
 public:
  GroupScheduler(ir::IRSchedule* ir_sch,
                 const std::shared_ptr<Graph::Group>& group,
                 const common::Target& target);

  void operator()();

 private:
  bool LoopAssign(ir::ScheduleBlockNode* source_node,
                  ir::ScheduleBlockNode* target_node);
  void DoComputeInline();
  void DoHorizontalLoopFusion();
  void DoVerticalLoopFusion();
  void AllocateStorage();

  int64_t NodePriority(const ir::ScheduleBlockNode* node) const;
  ir::ScheduleBlockNode* FindGlobalMasterNode() const;
  void UpdateBlockOrder();
  std::unordered_set<std::string> OutputTensorNames() const;

  bool IsKeepGraphDependency(Expr schedule_block,
                             Expr target_loop,
                             int insert_pos) const;
  bool MeetConditions(Expr schedule_block,
                      Expr target_loop,
                      int insert_pos) const;

 private:
  using FeasibleCondition = bool (GroupScheduler::*)(Expr schedule_block,
                                                     Expr target_loop,
                                                     int insert_pos) const;
  ir::IRSchedule* ir_sch_;
  const std::shared_ptr<Graph::Group>& group_;
  // const std::unordered_map<std::string, ir::Tensor>& tensor_map_;
  const common::Target& target_;
  std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph_;
  std::vector<FeasibleCondition> feasible_conditions_;
  std::map<std::vector<int>, ir::Expr> blocks_order_with_ctrl_stmt_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
