//------------------------------------------------------------------------------
/// \file
/// \brief      CommandLine class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"

#include <iostream>
#include <vector>
#include <regex>
#include <string>
#include <cassert>

class CommandLine {
public:
    CommandLine(int argc, char* argv[]) : argc(argc), argv(argv) {}

    void check(const std::vector<int>& indices,
               const std::regex& pattern) const {
        for (int idx : indices) {
            // Check if all indices are in range.
            assert(idx >= 1 && idx < argc);

            std::string argument = argv[idx];
            if (!std::regex_match(argument, pattern)) {
                errorOut(idx);
                return;
            }
        }
    }

    void check(int start, int end,
                const std::regex& pattern) const {
        assert(start >= 1 && end < argc && start <= end);

        for (int idx = start; idx <= end; ++idx) {
            std::string argument = argv[idx];
            if (!std::regex_match(argument, pattern)) {
                errorOut(idx);
            }
        }
    }

private:
    int argc;
    char** argv;

    void errorOut(int errorIndex) const {
        std::cout << std::endl;
        for (int i = 0; i < argc; ++i) {
            if (i < errorIndex) {
                std::cout << "\033[32m" << argv[i] << " ";
            } else if (i == errorIndex) {
                std::cout << "\e[5m\033[38;5;200m" << argv[i] << " ";
                std::cout << "\e[0m\033[0m" << "..." << std::endl;
                ERROR("Invalid argument");
            }
        }
    }
};
