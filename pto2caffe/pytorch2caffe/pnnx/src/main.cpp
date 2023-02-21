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

#include <stdio.h>

#if _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>
#include <vector>

#include <torch/script.h>

#include "ir.h"
#include "pass_level0.h"
#include "pass_level1.h"
#include "pass_level2.h"
#include "pass_level3.h"
#include "pass_level4.h"
#include "pass_level5.h"

#include "pass_ncnn.h"

static std::string get_basename(const std::string& path)
{
    return path.substr(0, path.find_last_of('.'));
}

static std::vector<std::string> parse_comma_string_array_list(char* s)
{
    std::vector<std::string> as;

    char* pch = strtok(s, ",");
    while (pch != NULL)
    {
        as.push_back(std::string(pch));

        pch = strtok(NULL, ",");
    }

    return as;
}

static std::vector<std::vector<int64_t> > parse_comma_int_array_list(char* s)
{
    std::vector<std::vector<int64_t> > aai;

    char* pch = strtok(s, "[]");
    while (pch != NULL)
    {
        // parse a,b,c
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1)
        {
            // ok we get array
            pch += nconsumed;

            std::vector<int64_t> ai;
            ai.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1)
            {
                pch += nconsumed;

                ai.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            // array end
            aai.push_back(ai);
        }

        pch = strtok(NULL, "[]");
    }

    return aai;
}

static void print_int64_array_list(const std::vector<std::vector<int64_t> >& list)
{
    for (size_t i = 0; i < list.size(); i++)
    {
        const std::vector<int64_t>& array = list[i];
        fprintf(stderr, "[");
        for (size_t j = 0; j < array.size(); j++)
        {
            fprintf(stderr, "%ld", array[j]);
            if (j != array.size() - 1)
                fprintf(stderr, ",");
        }
        fprintf(stderr, "]");
        if (i != list.size() - 1)
            fprintf(stderr, ",");
    }
}

static void print_string_list(const std::vector<std::string>& list)
{
    for (size_t i = 0; i < list.size(); i++)
    {
        fprintf(stderr, "%s", list[i].c_str());
        if (i + 1 != list.size())
            fprintf(stderr, ",");
    }
}

static void show_usage()
{
    fprintf(stderr, "Usage: pnnx [model.pt] [(key=value)...]\n");
    fprintf(stderr, "  pnnxparam=model.pnnx.param\n");
    fprintf(stderr, "  pnnxbin=model.pnnx.bin\n");
    fprintf(stderr, "  pnnxpy=model_pnnx.py\n");
    fprintf(stderr, "  ncnnparam=model.ncnn.param\n");
    fprintf(stderr, "  ncnnbin=model.ncnn.bin\n");
    fprintf(stderr, "  ncnnpy=model_ncnn.py\n");
    fprintf(stderr, "  optlevel=2\n");
    fprintf(stderr, "  device=cpu/gpu\n");
    fprintf(stderr, "  inputshape=[1,3,224,224],...\n");
    fprintf(stderr, "  inputshape2=[1,3,320,320],...\n");
#if _WIN32
    fprintf(stderr, "  customop=C:\\Users\\nihui\\AppData\\Local\\torch_extensions\\torch_extensions\\Cache\\fused\\fused.dll,...\n");
#else
    fprintf(stderr, "  customop=/home/nihui/.cache/torch_extensions/fused/fused.so,...\n");
#endif
    fprintf(stderr, "  moduleop=models.common.Focus,models.yolo.Detect,...\n");
    fprintf(stderr, "Sample usage: pnnx mobilenet_v2.pt inputshape=[1,3,224,224]\n");
    fprintf(stderr, "              pnnx yolov5s.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320] device=gpu moduleop=models.common.Focus,models.yolo.Detect\n");
}

static std::vector<std::string> ops_name_vec;
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        show_usage();
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            show_usage();
            return -1;
        }
    }

    std::string ptpath = std::string(argv[1]);

    std::string ptbase = get_basename(ptpath);

    std::string pnnxparampath = ptbase + ".pnnx.param";
    std::string pnnxbinpath = ptbase + ".pnnx.bin";
    std::string pnnxpypath = ptbase + "_pnnx.py";
    std::string ncnnparampath = ptbase + ".ncnn.param";
    std::string ncnnbinpath = ptbase + ".ncnn.bin";
    std::string ncnnpypath = ptbase + "_ncnn.py";
    int optlevel = 2;
    std::string device = "cpu";
    std::vector<std::vector<int64_t> > input_shapes;
    std::vector<std::vector<int64_t> > input_shapes2;
    std::vector<std::string> customop_modules;
    std::vector<std::string> module_operators;

    for (int i = 2; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];

        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        if (strcmp(key, "pnnxparam") == 0)
            pnnxparampath = std::string(value);
        if (strcmp(key, "pnnxbin") == 0)
            pnnxbinpath = std::string(value);
        if (strcmp(key, "pnnxpy") == 0)
            pnnxpypath = std::string(value);
        if (strcmp(key, "ncnnparam") == 0)
            ncnnparampath = std::string(value);
        if (strcmp(key, "ncnnbin") == 0)
            ncnnbinpath = std::string(value);
        if (strcmp(key, "ncnnpy") == 0)
            ncnnpypath = std::string(value);
        if (strcmp(key, "optlevel") == 0)
            optlevel = atoi(value);
        if (strcmp(key, "device") == 0)
            device = value;
        if (strcmp(key, "inputshape") == 0)
            input_shapes = parse_comma_int_array_list(value);
        if (strcmp(key, "inputshape2") == 0)
            input_shapes2 = parse_comma_int_array_list(value);
        if (strcmp(key, "customop") == 0)
            customop_modules = parse_comma_string_array_list(value);
        if (strcmp(key, "moduleop") == 0)
            module_operators = parse_comma_string_array_list(value);
    }

    // print options
    {
        fprintf(stderr, "pnnxparam = %s\n", pnnxparampath.c_str());
        fprintf(stderr, "pnnxbin = %s\n", pnnxbinpath.c_str());
        fprintf(stderr, "pnnxpy = %s\n", pnnxpypath.c_str());
        fprintf(stderr, "ncnnparam = %s\n", ncnnparampath.c_str());
        fprintf(stderr, "ncnnbin = %s\n", ncnnbinpath.c_str());
        fprintf(stderr, "ncnnpy = %s\n", ncnnpypath.c_str());
        fprintf(stderr, "optlevel = %d\n", optlevel);
        fprintf(stderr, "device = %s\n", device.c_str());
        fprintf(stderr, "inputshape = ");
        print_int64_array_list(input_shapes);
        fprintf(stderr, "\n");
        fprintf(stderr, "inputshape2 = ");
        print_int64_array_list(input_shapes2);
        fprintf(stderr, "\n");
        fprintf(stderr, "customop = ");
        print_string_list(customop_modules);
        fprintf(stderr, "\n");
        fprintf(stderr, "moduleop = ");
        print_string_list(module_operators);
        fprintf(stderr, "\n");
    }

    //     at::AutoNonVariableTypeMode nonVarTypeModeGuard(true);
    //     torch::autograd::AutoGradMode guard(false);

    for (auto m : customop_modules)
    {
        fprintf(stderr, "load custom module %s\n", m.c_str());
#if _WIN32
        HMODULE handle = LoadLibraryExA(m.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!handle)
        {
            fprintf(stderr, "LoadLibraryExA %s failed %s\n", m.c_str(), GetLastError());
        }
#else
        void* handle = dlopen(m.c_str(), RTLD_LAZY);
        if (!handle)
        {
            fprintf(stderr, "dlopen %s failed %s\n", m.c_str(), dlerror());
        }
#endif
    }

    std::vector<at::Tensor> input_tensors;
    for (auto shape : input_shapes)
    {
        at::Tensor t = torch::ones(shape);
        if (device == "gpu")
            t = t.cuda();

        input_tensors.push_back(t);
    }

    std::vector<at::Tensor> input_tensors2;
    for (auto shape : input_shapes2)
    {
        at::Tensor t = torch::ones(shape);
        if (device == "gpu")
            t = t.cuda();

        input_tensors2.push_back(t);
    }

    torch::jit::Module mod = torch::jit::load(ptpath);

    mod.eval();

    //     mod.dump(true, false, false);
    //     mod.dump(true, true, true);

    auto g = mod.get_method("forward").graph();

    //     g->dump();

    fprintf(stderr, "############# pass_level0\n");

    pnnx::pass_level0(mod, g, input_tensors, input_tensors2, module_operators, ops_name_vec);

    //     g->dump();

    fprintf(stderr, "############# pass_level1\n");

    pnnx::Graph pnnx_graph;
    pnnx::pass_level1(mod, g, pnnx_graph);

    //     g->dump();

    fprintf(stderr, "############# pass_level2\n");

    pnnx::pass_level2(pnnx_graph);

    pnnx_graph.save("debug.param", "debug.bin");

    if (optlevel >= 1)
    {
        fprintf(stderr, "############# pass_level3\n");

        pnnx::pass_level3(pnnx_graph);

        fprintf(stderr, "############# pass_level4\n");

        pnnx::pass_level4(pnnx_graph);
    }

    pnnx_graph.save("debug2.param", "debug2.bin");

    if (optlevel >= 2)
    {
        fprintf(stderr, "############# pass_level5\n");

        pnnx::pass_level5(pnnx_graph);
    }

    pnnx_graph.save(pnnxparampath, pnnxbinpath);

    pnnx_graph.python(pnnxpypath, pnnxbinpath);

    //     if (optlevel >= 2)
    {
        fprintf(stderr, "############# pass_ncnn\n");

        pnnx::pass_ncnn(pnnx_graph);

        pnnx_graph.ncnn(ncnnparampath, ncnnbinpath, ncnnpypath);
    }

    //     pnnx::Graph pnnx_graph2;

    //     pnnx_graph2.load("pnnx.param", "pnnx.bin");
    //     pnnx_graph2.save("pnnx2.param", "pnnx2.bin");

    return 0;
}


static pnnx::Graph pnnx_graph;
static torch::jit::Module model;
static std::map<std::string, at::Tensor> layer_output;
extern "C" {
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

// Model
int parse(char *ptPath, char *inputs_shape)
{
    std::string ptPathStr = std::string(ptPath);
    std::string device = "cpu";

    std::cout << "Parsing model: " << ptPathStr << std::endl;

    model = torch::jit::load(ptPathStr);
    model.eval();

    auto graph = model.get_method("forward").graph();

    std::string inputs_shape_str = inputs_shape;

    std::vector<std::string> shape_vec;
    SplitString(inputs_shape_str, shape_vec, "|");

    std::vector<at::Tensor> input_tensors;
    std::vector<std::vector<int64_t> > input_shapes = parse_comma_int_array_list(const_cast<char*>(shape_vec[0].data()));
    for (auto shape : input_shapes) {
        at::Tensor t = torch::ones(shape);
        if (device == "gpu")
            t = t.cuda();

        input_tensors.push_back(t);
    }

    std::vector<at::Tensor> input_tensors2;
    std::vector<std::vector<int64_t> > input_shapes2;
    if (shape_vec.size() >= 2) {
        input_shapes2 = parse_comma_int_array_list(const_cast<char*>(shape_vec[1].data()));
    }
    for (auto shape : input_shapes2) {
        at::Tensor t = torch::ones(shape);
        if (device == "gpu")
            t = t.cuda();

        input_tensors2.push_back(t);
    }

    std::vector<std::string> module_operators;

    pnnx::pass_level0(model, graph, input_tensors, input_tensors2, module_operators, ops_name_vec);
    pnnx::pass_level1(model, graph, pnnx_graph);
    pnnx::pass_level2(pnnx_graph);
    pnnx::pass_level3(pnnx_graph);
    pnnx::pass_level4(pnnx_graph);
    pnnx::pass_level5(pnnx_graph);

    std::cout << "Parsing Done" << std::endl;
    return 0;
}

int model_forward(char *input_buf, char *inputs_shape)
{
    std::string device = "cpu";

    auto graph = model.get_method("forward").graph();

    std::string inputs_shape_str = inputs_shape;
    std::vector<std::string> shape_vec;
    SplitString(inputs_shape_str, shape_vec, "|");

    std::vector<at::Tensor> input_tensors;
    std::vector<std::vector<int64_t> > input_shapes = parse_comma_int_array_list(const_cast<char*>(shape_vec[0].data()));
    for (auto shape : input_shapes) {
        at::Tensor t = torch::ones(shape);
        memcpy(t.cpu().data_ptr(), input_buf, 4 * sizeof(torch::kU8) * t.numel());
        if (device == "gpu")
            t = t.cuda();

        input_tensors.push_back(t);
    }

    std::vector<torch::jit::IValue> inputs;
    for (size_t i = 0; i < input_tensors.size(); i++) {
        const at::Tensor& it = input_tensors[i];

        inputs.push_back(it);
        graph->inputs()[1 + i]->setType(c10::TensorType::create(it));
    }

    auto outputs = model.forward(inputs).toTuple();
    int index = 0;
    for (auto e : outputs->elements()) {
        layer_output[ops_name_vec[index]] = e.toTensor();
        index++;
    }

    return index;
}

//Get Output
int get_ops_output(char* op_name, char* buf)
{
    if (layer_output[op_name].numel() > 0) {
        memcpy(buf, layer_output[op_name].data_ptr(), layer_output[op_name].data().numel() * layer_output[op_name].data().itemsize());
    }

    return layer_output[op_name].numel();
}

// Operator
unsigned int get_ops_len()
{
    return (int)pnnx_graph.ops.size();
}

const char *get_ops_type(unsigned int ops_no)
{
    return pnnx_graph.ops[ops_no]->type.c_str();
}

const char *get_ops_name(unsigned int ops_no)
{
    return pnnx_graph.ops[ops_no]->name.c_str();
}

// Input Operand
unsigned int get_inputs_len(unsigned int ops_no)
{
    return pnnx_graph.ops[ops_no]->inputs.size();
}

const char *get_input_name(unsigned int ops_no, unsigned int input_no)
{
    return pnnx_graph.ops[ops_no]->inputs[input_no]->name.c_str();
}

const char *get_ops_input_shape(unsigned int ops_no, unsigned int input_no)
{
    std::string input_shape = "";

    if ((input_no + 1) > pnnx_graph.ops[ops_no]->inputs.size())
        return input_shape.c_str();

    for (auto dim : pnnx_graph.ops[ops_no]->inputs[input_no]->shape) {
        input_shape = input_shape + std::to_string(dim) + ",";
    }

    if (input_shape.length() > 0)
        input_shape.pop_back();

    return input_shape.c_str();
}

// Output Operand
unsigned int get_outputs_len(unsigned int ops_no)
{
    return pnnx_graph.ops[ops_no]->outputs.size();
}

const char *get_output_name(unsigned int ops_no, unsigned int output_no)
{
    return pnnx_graph.ops[ops_no]->outputs[output_no]->name.c_str();
}

const char *get_ops_output_shape(unsigned int ops_no, unsigned int output_no)
{
    std::string output_shape = "";

    if ((output_no + 1) > pnnx_graph.ops[ops_no]->outputs.size())
        return output_shape.c_str();

    for (auto dim : pnnx_graph.ops[ops_no]->outputs[output_no]->shape) {
        output_shape = output_shape + std::to_string(dim) + ",";
    }
    if (output_shape.length() > 0)
        output_shape.pop_back();

    return output_shape.c_str();
}


// Attribute
unsigned int get_ops_attrs_len(unsigned int ops_no)
{
    return pnnx_graph.ops[ops_no]->attrs.size();
}

const char *get_ops_attrs_names(unsigned int ops_no)
{
    std::string attrs_names;
    for (const auto& attr_item : pnnx_graph.ops[ops_no]->attrs) {
        attrs_names = attrs_names + attr_item.first + ",";
    }

    if (attrs_names.length() > 0)
        attrs_names.pop_back();

    return attrs_names.c_str();
}

unsigned int get_ops_attr_type(unsigned int ops_no, char *attr_name)
{
    std::string name = attr_name;
    if (pnnx_graph.ops[ops_no]->attrs.find(name) == pnnx_graph.ops[ops_no]->attrs.end())
        return -1;
    else
        return pnnx_graph.ops[ops_no]->attrs[name].type;
}

const char *get_ops_attr_shape(unsigned int ops_no, char *attr_name)
{
    std::string attr_shape;
    std::string name = attr_name;

    for (auto dim : pnnx_graph.ops[ops_no]->attrs[name].shape) {
        attr_shape = attr_shape + std::to_string(dim) + ",";
    }
    if (attr_shape.length() > 0)
        attr_shape.pop_back();

    return attr_shape.c_str();
}

unsigned int get_ops_attr_data_size(unsigned int ops_no, char *attr_name)
{
    std::string name = attr_name;

    if (pnnx_graph.ops[ops_no]->attrs.find(name) == pnnx_graph.ops[ops_no]->attrs.end())
        return 0;
    else
        return pnnx_graph.ops[ops_no]->attrs[name].data.size();
}

unsigned int get_ops_attr(unsigned int ops_no, char *attr_name, char* buf)
{
    std::string name = attr_name;

    memcpy(buf, pnnx_graph.ops[ops_no]->attrs[name].data.data(), pnnx_graph.ops[ops_no]->attrs[name].data.size());

    return pnnx_graph.ops[ops_no]->attrs.size();
}

// Parameter
const char *get_ops_param(unsigned int ops_no)
{
    std::string param_str;

    for (const auto& param_item : pnnx_graph.ops[ops_no]->params) {
        param_str = param_str + param_item.first + "=";
        const pnnx::Parameter& param = param_item.second;

        if (param.type == 0) {
            param_str += "None";
            param_str += "@None";
        }
        if (param.type == 1) {
            if (param.b)
                param_str += "True";
            else
                param_str += "False";
            param_str += "@bool";
        }
        if (param.type == 2) {
            param_str += std::to_string(param.i);
            param_str += "@int";
        }
        if (param.type == 3) {
            param_str += std::to_string(param.f);
            param_str += "@float";
        }
        if (param.type == 4) {
            param_str += param.s;
            param_str += "@string";
        }
        if (param.type == 5) {
            for (size_t i = 0; i < param.ai.size(); i++) {
                param_str += std::to_string(param.ai[i]);
                if (i + 1 != param.ai.size())
                    param_str += ",";
            }
            param_str += "@[int]";
        }
        if (param.type == 6) {
            for (size_t i = 0; i < param.af.size(); i++) {
                param_str += std::to_string(param.af[i]);
                if (i + 1 != param.af.size())
                    param_str += ",";
            }
            param_str += "@[float]";
        }
        if (param.type == 7) {
            for (size_t i = 0; i < param.af.size(); i++) {
                param_str += param.as[i].c_str();
                if (i + 1 != param.af.size())
                    param_str += ",";
            }
            param_str += "@[string]";
        }
        param_str += "|";
    }
    if (param_str.length() > 0)
        param_str.pop_back();

    return param_str.c_str();
}
}

