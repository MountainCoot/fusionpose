#ifndef GENERAL_H
#define GENERAL_H

#include <string>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

void keep_max_n_files(const std::string& file_name, int n = 4) {
    // Extract file extension and base name
    std::string ext = fs::path(file_name).extension().string();
    std::string file_name_strp = fs::path(file_name).stem().string();
    std::string parent_dir = fs::path(file_name).parent_path().string();
    
    // If file_name doesn't exist, return early
    if (!fs::exists(file_name)) {
        return;
    }

    // Iterate backwards to delete the n-th file and rename the others
    for (int i = n - 1; i >= 0; --i) {
        std::string file_name_i = (i > 0) ? parent_dir + "/" + file_name_strp + "_" + std::to_string(i) + ext : file_name;

        // Remove the file at the limit if it exists
        if (i == n - 1 && fs::exists(file_name_i)) {
            fs::remove(file_name_i);
        } else {
            // Rename or copy the previous file to the next version
            std::string file_name_i_1 = parent_dir + "/" + file_name_strp + "_" + std::to_string(i + 1) + ext;
            if (fs::exists(file_name_i)) {
                fs::copy(file_name_i, file_name_i_1, fs::copy_options::overwrite_existing);
            }
        }
    }
}

#endif // GENERAL_H