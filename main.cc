#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"

void initialise() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Running on CUDA device " << prop.name << std::endl;
  int value;
  cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  std::cout << "  - maximum shared memory per block: " << value / 1024 << " kB" << std::endl;
}

bool read_next_event(std::istream& input, std::vector<PseudoJet>& particles) {
  // clear the output buffer
  particles.clear();

  // clear the input status flags
  input.clear();

  // skip comments and empty lines
  while (input.peek() == '#' or input.peek() == '\n') {
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  // read the input one line at a time
  int i = 0;
  std::string buffer;
  while (std::getline(input, buffer).good()) {
    std::istringstream line(buffer);

    // read the four elements
    double px, py, pz, E;
    line >> px >> py >> pz >> E;
    //std::cout << "reading: " << px << ", " << py << ", " << pz << ", " << E << std::endl;

    if (line.fail()) {
      //std::cout << "no more particles" << std::endl;
      // check for a comment or empty line
      if (not buffer.empty() and buffer[0] != '#') {
        throw std::runtime_error("Error while parsing particles:\n" + buffer);
      }
      break;
    }

    //std::cout << "found a particle" << std::endl;
    particles.push_back({i++, false, px, py, pz, E});
  }

  // return false if there was no event to read
  return (not particles.empty());
}

void print_jets(std::vector<PseudoJet> const& jets, bool cartesian = false) {
  std::cout << std::fixed << std::setprecision(8);
  int i = 0;
  for (auto const& jet : jets) {
    if (cartesian) {
      // print px, py, pz, E
      std::cout << std::setw(5) << i++ << std::setw(16) << jet.px << std::setw(16) << jet.py << std::setw(16) << jet.pz
                << std::setw(16) << jet.E << std::endl;
    } else {
      // print eta, phi, pT
      double pT = std::hypot(jet.px, jet.py);
      double phi = atan2(jet.py, jet.px);
      while (phi > 2 * M_PI) {
        phi -= 2 * M_PI;
      }
      while (phi < 0.) {
        phi += 2 * M_PI;
      }
      double effective_m2 = std::max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - pT * pT);
      double E_plus_pz = jet.E + std::abs(jet.pz);
      double eta = 0.5 * std::log((pT * pT + effective_m2) / (E_plus_pz * E_plus_pz));
      if (jet.pz > 0) {
        eta = -eta;
      }
      std::cout << std::setw(5) << i++ << std::setw(16) << eta << std::setw(16) << phi << std::setw(16) << pT
                << std::endl;
    }
  }
  std::cout << std::endl;
}

int main(int argc, const char* argv[]) {
  double ptmin = 0.0;          // GeV
  double r = 1.0;              // clustering radius
  Scheme scheme = Scheme::Kt;  // recombination scheme
  bool sort = true;
  bool cartesian = false;
  int repetitions = 1;
  bool output_csv = false;
  std::string filename;  // read data from file instead of standard input

  for (unsigned int i = 1; i < argc; ++i) {
    // --ptmin, -p
    if (std::strcmp(argv[i], "--ptmin") == 0 or std::strcmp(argv[i], "-p") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtod(argv[i], &stop);
      if (stop != argv[i] and arg >= 0.) {
        ptmin = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // -r, -R
        if (std::strcmp(argv[i], "-r") == 0 or std::strcmp(argv[i], "-R") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtod(argv[i], &stop);
      if (stop != argv[i] and arg >= 0) {
        r = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // --repeat, -repeat
        if (std::strcmp(argv[i], "--repeat") == 0 or std::strcmp(argv[i], "-repeat") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtol(argv[i], &stop, 0);
      if (stop != argv[i] and arg >= 0) {
        repetitions = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // --sort, -s
        if (std::strcmp(argv[i], "--sort") == 0 or std::strcmp(argv[i], "-s") == 0) {
      sort = true;
    } else

        // --cartesian
        if (std::strcmp(argv[i], "--cartesian") == 0) {
      cartesian = true;
    } else

        // --polar
        if (std::strcmp(argv[i], "--polar") == 0) {
      cartesian = false;
    } else

        // --kt, -kt
        if (std::strcmp(argv[i], "--kt") == 0 or std::strcmp(argv[i], "-kt") == 0) {
      scheme = Scheme::Kt;
    } else

        // --anti-kt, -antikt
        if (std::strcmp(argv[i], "--anti-kt") == 0 or std::strcmp(argv[i], "-antikt") == 0) {
      scheme = Scheme::AntiKt;
    } else

        // --cambridge-aachen, -cam
        if (std::strcmp(argv[i], "--cambridge-aachen") == 0 or std::strcmp(argv[i], "-cam") == 0) {
      scheme = Scheme::CambridgeAachen;
    } else

        // --file, -f
        if (std::strcmp(argv[i], "--file") == 0 or std::strcmp(argv[i], "-f") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      filename = argv[i];
    } else

        if (std::strcmp(argv[i], "--csv") == 0 or std::strcmp(argv[i], "-csv") == 0) {
      output_csv = true;
    } else

    // unknown option
    {
      std::cerr << "Unrecognized option " << argv[i] << std::endl;
      return 1;
    }
  }

  // initialise the GPU
  initialise();

  // open an input file
  std::ifstream input(filename, std::ios_base::in);

  std::vector<PseudoJet> particles;
  std::vector<PseudoJet> jets;

  while (read_next_event(filename.empty() ? std::cin : input, particles)) {
    std::cout << "found " << particles.size() << " particles" << std::endl;

    // allocate GPU memory for the input particles
    PseudoJet* particles_d;
    cudaCheck(cudaMalloc(&particles_d, sizeof(PseudoJet) * particles.size()));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    double sum = 0.;
    double sum2 = 0.;
    for (int step = 0; repetitions == 0 or step < repetitions; ++step) {
      cudaCheck(cudaEventRecord(start));

      // copy the input to the GPU
      cudaCheck(cudaMemcpy(particles_d, particles.data(), sizeof(PseudoJet) * particles.size(), cudaMemcpyDefault));

      // run the clustering algorithm and measure its running time
      cluster(particles_d, particles.size(), scheme, r);

      // copy the clustered jets back to the CPU
      jets.resize(particles.size());
      cudaCheck(cudaMemcpy(jets.data(), particles_d, sizeof(PseudoJet) * jets.size(), cudaMemcpyDefault));

      cudaCheck(cudaEventRecord(stop));
      cudaCheck(cudaEventSynchronize(stop));

      float milliseconds;
      cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
      sum += milliseconds;
      sum2 += milliseconds * milliseconds;

      // remove the unused elements and the jets with pT < pTmin
      auto last = std::remove_if(jets.begin(), jets.end(), [ptmin](auto const& jet) {
        return (not jet.isJet) or (jet.px * jet.px + jet.py * jet.py < ptmin * ptmin);
      });
      jets.erase(last, jets.end());

      if (not output_csv) {
        if (ptmin > 0.) {
          std::cout << "found " << jets.size() << " jets above " << ptmin << " GeV in " << milliseconds << " ms"
                    << std::endl;
        } else {
          std::cout << "found " << jets.size() << " jets in " << milliseconds << " ms" << std::endl;
        }
      }

      // optionally, sort the jets by decreasing pT
      if (sort) {
        std::sort(jets.begin(), jets.end(), [](auto const& a, auto const& b) {
          return (a.px * a.px + a.py * a.py > b.px * b.px + b.py * b.py);
        });
      }
    }

    // free GPU memory
    cudaCheck(cudaFree(particles_d));

    print_jets(jets, cartesian);

    std::cout << std::defaultfloat;

    if (not output_csv) {
      std::cout << "clustered " << particles.size() << " particles into " << jets.size() << " jets above " << ptmin
                << " GeV";
    } else {
      std::out << particles.size() << ',' << jets.size() << ',';
    }
    std::cout << std::fixed;
    double mean = sum / repetitions;
    int precision;
    if (repetitions > 1) {
      double sigma = std::sqrt((sum2 - sum * sum / repetitions) / (repetitions - 1));
      precision = std::max((int)-std::log10(sigma / 2.) + 1, 0);
      precision = std::cout.precision(precision);
      if (not output_csv) {
        std::cout << " in " << mean << " +/- " << sigma << " ms" << std::endl;
      } else {
        std::cout << mean << ',' << sigma << std::endl;
      }
    } else {
      precision = std::cout.precision(1);
      if (not output_csv) {
        std::cout << " in " << mean << " ms" << std::endl;
      } else {
        std::cout << mean << ',' << sigma << std::endl;
      }
    }
    std::cout.precision(precision);
    std::cout << std::defaultfloat;
  }

  return 0;
}
