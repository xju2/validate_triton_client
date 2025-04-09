#include "TritonClientTool.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace tc = triton::client;

std::vector<float> rn50_preprocess(const std::string &full_path)
{
  // Read raw binary data
  std::ifstream bin_file(full_path);
  int height = 224, width = 224, channels = 3;
  int total_size = height * width * channels;
  std::vector<float> data;
  data.reserve(total_size);
  float val;
  while (bin_file) {
    bin_file >> val;
    if (!bin_file.fail()) data.push_back(val);
  }
  return data;
}


// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    int opt;
    bool help = false;
    bool verbose = false;
    std::string input_file = "img1.bin";
    std::string url("localhost:8001");
    std::string model_name("resnet50");
    float tol = 0.01;
    
    while ((opt = getopt(argc, argv, "vhu:i:t:")) != -1) {
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            case 'u':
                url = optarg;
                break;
            case 'i':
                input_file = optarg;
                break;
            case 't':
                tol = std::stof(optarg);
                break;
            case 'h':
                help = true;
            default:
                fprintf(stderr, "Usage: %s [-hv] [-u URL]\n", argv[0]);
                if (help) {
                    std::cerr << " -u: url of server" << std::endl;
                    std::cerr << " -i: input file" << std::endl;
                    std::cerr << " -v: verbose" << std::endl;
                }
            exit(EXIT_FAILURE);
        }
    }

    std::cout <<"Reading input file: " << input_file << std::endl;

    uint32_t client_timeout = 0;
    std::string model_version = "";
    auto m_client = std::make_unique<TritonClientTool>(
        model_name, url, model_version, client_timeout, verbose);

    m_client->ClearInput();

    std::vector<float> inputValues = rn50_preprocess(input_file);
    std::cout << "total input entries: " << inputValues.size() << std::endl;
    std::vector<int64_t> inputShape{3, 224, 224};

    // veryfy input values
    std::vector<float> expectedInputValues{
        1.015926, 1.0330508, 1.015926, 1.015926, 1.015926
    };
    auto vec_to_string = [](const std::vector<float>& vec, size_t max_size = 5) -> std::string {
        if (vec.size() < max_size) {
            return "[Wrong size]";
        }
        std::string result = "[";
        for (size_t i = 0; i < max_size; ++i) {
            result += std::to_string(vec[i]);
            if (i != vec.size() - 1) {
                result += " ";
            }
        }
        result += "]";
        return result;
    };
    auto all_close = [](const std::vector<float>& a, const std::vector<float>& b, float tolerance) -> bool {
        for (size_t i = 0; i < b.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    };
    std::cout << "Input values: " << vec_to_string(inputValues) << std::endl;
    std::cout << "Expected input: " << vec_to_string(expectedInputValues) << std::endl;
    if (!all_close(inputValues, expectedInputValues, tol)) {
        std::cerr << "Input values do not match expected values within " << tol << std::endl;
        return -1;
    } else {
        std::cout << "Input values match expected values within " << tol << std::endl;
    }

    m_client->AddInput<float>("input__0", inputShape, inputValues);
    std::vector<float> outProbs;
    std::vector<int64_t> outShape{1, 1000};
    m_client->GetOutput<float>("output__0", outProbs, outShape);

    std::vector<float> expectedOutputValues{
        -0.25864124, 4.3533406, -2.0574622, -1.9991533, -3.178356
    };

    std::cout << "Output values: " << vec_to_string(outProbs) << std::endl;
    std::cout << "Expected output: " << vec_to_string(expectedOutputValues) << std::endl;
    if (!all_close(outProbs, expectedOutputValues, tol)) {
        std::cerr << "Output values do not match expected values within " << tol << std::endl;
        return -1;
    } else {
        std::cout << "Output values match expected values within " << tol << std::endl;
    }

    return 0;
}

