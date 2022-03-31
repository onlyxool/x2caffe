// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "shape_inference.h"

namespace pnnx {

void shape_inference(const torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>& graph, const std::vector<at::Tensor>& input_tensors, const std::vector<at::Tensor>& input_tensors2, std::vector<std::string>& ops_name_vec)
{
    // collect all intermediate output tensors
    std::vector<torch::jit::Value*> values;
    int index = 0;
    for (const auto& n : graph->nodes())
    {
        for (const auto& on : n->outputs())
        {
            auto tensor_type = on->type()->cast<torch::jit::TensorType>();
            if (!tensor_type)
                continue;

            values.push_back(on);
            index++;
            if (n->kind() == c10::prim::CallMethod) {
                std::deque<std::string> module_names;
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name))
                    {
                        module_names.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrapped_name;
                auto sub_mod = mod;
                for (auto module_name : module_names)
                {
                    if (wrapped_name.size() > 0)
                        wrapped_name = wrapped_name + "." + module_name;
                    else
                        wrapped_name = module_name;
                    sub_mod = sub_mod.attr(module_name).toModule();
                }
                ops_name_vec.push_back(wrapped_name);
            } else {
                ops_name_vec.push_back("None");
            }
        }
    }

    // set new graph output
    auto old_output = graph->outputs()[0];

    torch::jit::Node* new_return_node = graph->createTuple(at::ArrayRef<torch::jit::Value*>(values));

    graph->appendNode(new_return_node);

    graph->eraseOutput(0);
    graph->registerOutput(new_return_node->outputs()[0]);

    // inference for all tensors
    std::vector<torch::jit::IValue> inputs;
    for (size_t i = 0; i < input_tensors.size(); i++)
    {
        const at::Tensor& it = input_tensors[i];

        inputs.push_back(it);
        graph->inputs()[1 + i]->setType(c10::TensorType::create(it));
    }

    auto outputs = mod.copy().forward(inputs).toTuple();

    if (input_tensors2.empty())
    {
        // assign shape info
        int index = 0;
        for (auto e : outputs->elements())
        {
            values[index]->setType(c10::TensorType::create(e.toTensor()));

            index++;
        }
    }
    else
    {
        std::vector<torch::jit::IValue> inputs2;
        for (size_t i = 0; i < input_tensors2.size(); i++)
        {
            const at::Tensor& it = input_tensors2[i];

            inputs2.push_back(it);
            graph->inputs()[1 + i]->setType(c10::TensorType::create(it));
        }

        auto outputs2 = mod.copy().forward(inputs2).toTuple();

        fprintf(stderr, "assign dynamic shape info\n");

        // assign dynamic shape info
        for (size_t i = 0; i < input_tensors.size(); i++)
        {
            auto type1 = c10::TensorType::create(input_tensors[i]);
            auto type2 = c10::TensorType::create(input_tensors2[i]);

            std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
            std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

            for (size_t i = 0; i < sizes1.size(); i++)
            {
                if (sizes1[i] == sizes2[i])
                    continue;

                sizes1[i] = c10::ShapeSymbol::fromStaticSize(-1);
            }

            auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));

            graph->inputs()[1 + i]->setType(finaltype);
        }

        int index = 0;
        for (auto e : outputs->elements())
        {
            auto type1 = c10::TensorType::create(e.toTensor());
            auto type2 = c10::TensorType::create(outputs2->elements()[index].toTensor());

            std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
            std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

            for (size_t i = 0; i < sizes1.size(); i++)
            {
                if (sizes1[i] == sizes2[i])
                    continue;

                sizes1[i] = c10::ShapeSymbol::fromStaticSize(-1);
            }

            auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));

            values[index]->setType(finaltype);

            index++;
        }
    }

    // restore old graph output
    graph->eraseOutput(0);
    graph->registerOutput(old_output);

    new_return_node->removeAllInputs();
    new_return_node->destroy();
}

} // namespace pnnx
