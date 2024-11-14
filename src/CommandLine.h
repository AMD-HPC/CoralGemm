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

//------------------------------------------------------------------------------
/// \brief
///     A utility class to manage and validate command-line arguments.
///
class CommandLine {
public:
    ///
    /// \brief
    ///     Creates a CommandLine object.
    ///
    /// \param[in] argc
    ///     the number of command-line arguments
    ///
    /// \param[in] argv
    ///     the array of command-line arguments
    ///
    CommandLine(int argc, char* argv[]) : argc(argc), argv(argv) {}

    ///
    /// \brief
    ///     Validates specified command-line arguments against a regex pattern.
    ///
    /// \param[in] indices
    ///     a vector of indices of arguments to be checked
    ///
    /// \param[in] pattern
    ///     the regular expression pattern each argument should match
    ///
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

    ///
    /// \brief
    ///     Validates specified command-line arguments against a regex pattern.
    ///
    /// \param[in] start
    ///     the starting index (inclusive) of the arguments to be checked
    ///
    /// \param[in] end
    ///     the ending index (inclusive) of the arguments to be checked
    ///
    /// \param[in] pattern
    ///     the regular expression pattern each argument should match
    ///
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
    int argc;    ///< The number of command-line arguments.
    char** argv; ///< The array of command-line arguments.

    ///
    /// \brief
    ///     Prints an error message for an invalid argument end errors out.
    ///
    /// \param[in] error_index
    ///     The index of the invalid argument in argv.
    ///
    void errorOut(int error_index) const {
        std::cout << std::endl;
        for (int i = 0; i < argc; ++i) {
            if (i < error_index) {
                std::cout << "\033[32m" << argv[i] << " ";
            } else if (i == error_index) {
                std::cout << "\e[5m\033[38;5;200m" << argv[i] << " ";
                std::cout << "\e[0m\033[0m" << "..." << std::endl;
                ERROR("Invalid argument");
            }
        }
    }
};
