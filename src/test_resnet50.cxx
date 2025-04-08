#include "TritonClientTool.hpp"

namespace tc = triton::client;

std::vector<float> rn50_preprocess(const std::string &full_path)
{
  // Read raw binary data
  std::ifstream bin_file(full_path, std::ios::binary);
  int height = 224, width = 224, channels = 3;
  int total_size = height * width * channels;
  std::vector<float> data(total_size);
  bin_file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
 return input_tensor_values;
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
    
    while ((opt = getopt(argc, argv, "vhu:i:")) != -1) {
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

    bool verbose = false;
    uint32_t client_timeout = 0;
    std::string model_version = "";
    auto m_client = std::make_unique<TritonClientTool>(
        model_name, url, model_version, client_timeout, verbose);

    m_client->ClearInput();

    std::vector<float> inputValues = rn50_preprocess(input_file);
    std::vector<int64_t> inputShape{1, 3, 224, 224};

    m_client->AddInput<float>("input__0", inputShape, inputValues);
    std::vector<float> outProbs;
    std::vector<int64_t> outShape{1, 1000};
    m_client->GetOutput<int64_t>("LABELS", outProbs, outShape);

    std::string expected_values = "[-0.25864124  4.3533406  -2.0574622  -1.9991533  -3.178356]";
    std::string output_values = "[";
    for (size_t i = 0; i < 5; ++i) {
        output_values += std::to_string(outProbs[i]);
        if (i != outProbs.size() - 1) {
            output_values += " ";
        }
    }
    output_values += "]";
    std::cout << "Output: " << output_values << std::endl;
    std::cout << "Expected: " << expected_values << std::endl;
    return 0;
}

