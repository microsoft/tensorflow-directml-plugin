set(CMAKE_FOLDER ThirdParty)

include(FetchContent)

# Google Abseil C++ Library
FetchContent_Declare(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG 20220623.0
)

# Google Protobuf
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v3.19.3
    SOURCE_SUBDIR cmake
)
set(protobuf_BUILD_TESTS OFF)

# TensorFlow python package
if(WIN32)
    # To update the package version to a nightly build, go to https://pypi.org/project/tf-nightly-intel/#files and copy the
    # URLs and SHA256 hashes for the cp37 versions.
    FetchContent_Declare(
        tensorflow_whl
        URL https://files.pythonhosted.org/packages/81/95/d4cb742c278bbf84f9361faa5de846d6ac05c1f88327bc98d22f1c25994e/tf_nightly_intel-2.10.0.dev20220728-cp37-cp37m-win_amd64.whl
        URL_HASH SHA256=4f33660b3dbf0b22f069c27e332e3fa00a6d22b0fd5e5fd4015c8185bf3e9227
    )

    FetchContent_Declare(
        tensorflow_framework
        URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.9.1.zip
        URL_HASH SHA256=56fa7e84c984144311563b405bd0e565568f3cbe2c15ac4713e9c8174a02b00f
    )
else()
    # To update the package version to a nightly build, go to https://pypi.org/project/tf-nightly-cpu/#files and copy the
    # URLs and SHA256 hashes for the cp37 versions.
    FetchContent_Declare(
        tensorflow_whl
        URL https://files.pythonhosted.org/packages/09/19/fec07df7f1a2a58e43993883cf78433e467b708783375ab3ef777da78876/tf_nightly_cpu-2.10.0.dev20220728-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        URL_HASH SHA256=06d0e9cbd8b41eb45b120822a765c584ecf6d8e7f6b888469fa9e87643d4c479
    )

    FetchContent_Declare(
        tensorflow_framework
        URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.9.1.tar.gz
        URL_HASH SHA256=7f510d2546e3db5e10b1d119463689003d587842892ad3c95bca51cf82820173
    )
endif()

# DirectX-Headers repo
FetchContent_Declare(
    directx_headers
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers
    GIT_TAG d49ae12ab350b20468a9667bad700f3227cd3f7a
)

# DirectML Redistributable NuGet package
FetchContent_Declare(
    directml_redist
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.9.0
    URL_HASH SHA256=052224190fa5db862694c29799c8eb7945191f5d0046b736f9864bd98417da14
)

# DirectMLX helper library
FetchContent_Declare(
    directmlx
    URL https://raw.githubusercontent.com/microsoft/DirectML/9d051781349b1c706695b148e2e31eb9c50de226/Libraries/DirectMLX.h
    URL_HASH SHA256=81bd6b556356b438e84ec681fe975e87d94b4363ee95caa2e2401a237fed0ee3
    DOWNLOAD_NO_EXTRACT TRUE
)

# WinPixEventRuntime NuGet package
FetchContent_Declare(
    pix_event_runtime
    URL https://www.nuget.org/api/v2/package/WinPixEventRuntime/1.0.210209001
    URL_HASH SHA256=ee0af78308ea90c31b0c2a0c8814d2bef994e4cbfb5ae6c5b98b50c7fd98e1bc
)

# Download and extract dependencies.
FetchContent_MakeAvailable(
    abseil
    protobuf
    tensorflow_whl
    tensorflow_framework
    directx_headers
    directml_redist 
    directmlx
    pix_event_runtime
)

# The DirectX-Headers target assumes dependent targets include headers with the directx prefix 
# (e.g. <directx/d3d12.h>). However, directml.h unconditionally includes "d3d12.h"; this works on
# Windows with the SDK installed, but WSL builds need to resolve this include correctly.
target_include_directories(DirectX-Headers INTERFACE ${directx_headers_SOURCE_DIR}/include/directx)

# Target to add DirectML redist headers to the include path.
add_library(directml_headers INTERFACE)
target_include_directories(directml_headers INTERFACE ${directml_redist_SOURCE_DIR}/include)
add_library(directml_redist::headers ALIAS directml_headers)

# Target to add DirectMLX headers to the include path.
add_library(directmlx_headers INTERFACE)
target_include_directories(directmlx_headers INTERFACE ${directmlx_SOURCE_DIR})
add_library(directmlx::headers ALIAS directmlx_headers)

# Target (for convenience) that adds both DirectML and DirectMLX headers to the include path.
add_library(directml_all_headers INTERFACE)
target_link_libraries(directml_all_headers INTERFACE directml_headers directmlx_headers)
add_library(directml::headers ALIAS directml_all_headers)

# Target to add WinPixEventRuntime headers to the include path.
add_library(pix_headers INTERFACE)
target_include_directories(pix_headers INTERFACE ${pix_event_runtime_SOURCE_DIR}/include)
add_library(pix_event_runtime::headers ALIAS pix_headers)

# Location to generate .pb.h/.pb.cc files from the packaged TF .proto files. The TF wheel
# comes with pre-generated .pb.h files, but we need the source files (not included) when linking.
# This script uses the protobuf dependency above to regenerate C++ code for the .proto files 
# needed by the TFDML plugin. The .pb.h files and protobuf headers in the TF wheel are intentionally
# deleted to avoid accidental usage, since their protobuf version will not likely match the copy
# of protobuf above.
set(tensorflow_generated_protobuf_dir ${tensorflow_whl_BINARY_DIR}/proto)
set(tensorflow_include_dir ${tensorflow_whl_SOURCE_DIR}/tensorflow/include)
file(GLOB_RECURSE tensorflow_whl_pb_h_files ${tensorflow_include_dir}/**/*.pb.h)
if(tensorflow_whl_pb_h_files)
    file(REMOVE ${tensorflow_whl_pb_h_files})
endif()
file(REMOVE_RECURSE ${tensorflow_include_dir}/google/protobuf)

# We delete all abseil files from the TensorFlow include directory in order to avoid accidental usage.
# We don't want to depend on the TensorFlow abseil includes since it would mean that our binary would
# always need to have the same abseil version.
file(GLOB_RECURSE tensorflow_whl_absl_files ${tensorflow_include_dir}/absl/**/*.h)
if(tensorflow_whl_absl_files)
    file(REMOVE ${tensorflow_whl_absl_files})
endif()
file(REMOVE_RECURSE ${tensorflow_include_dir}/google/protobuf)

add_library(tensorflow_protos STATIC)
target_link_libraries(tensorflow_protos INTERFACE libprotobuf)
target_include_directories(
    tensorflow_protos
    PRIVATE
    $<TARGET_PROPERTY:libprotobuf,INCLUDE_DIRECTORIES>
    PUBLIC
    ${tensorflow_generated_protobuf_dir}
)

# Introduces a command to generate C++ code for a .proto file in the TF wheel.
function(tf_proto_cpp proto_path)
    cmake_path(GET proto_path STEM proto_stem)
    cmake_path(GET proto_path PARENT_PATH proto_parent_dir)
    cmake_path(SET proto_generated_h ${tensorflow_generated_protobuf_dir}/${proto_parent_dir}/${proto_stem}.pb.h)
    cmake_path(SET proto_generated_cc ${tensorflow_generated_protobuf_dir}/${proto_parent_dir}/${proto_stem}.pb.cc)

    add_custom_command(
        OUTPUT 
            ${proto_generated_h} 
            ${proto_generated_cc}
        COMMAND 
            protobuf::protoc 
            --proto_path=${tensorflow_include_dir}
            --cpp_out=${tensorflow_generated_protobuf_dir} 
            ${proto_path}
        DEPENDS
            ${tensorflow_include_dir}/${proto_path}
        COMMENT
            "Generating C++ code for ${proto_path}"
    )

    target_sources(tensorflow_protos PRIVATE ${proto_generated_h} ${proto_generated_cc})
endfunction()

# Generate the necessary .proto files in the TF wheel (performed at build time).
tf_proto_cpp(tensorflow/core/profiler/protobuf/xplane.proto)
tf_proto_cpp(tensorflow/core/framework/graph.proto)
tf_proto_cpp(tensorflow/core/framework/function.proto)
tf_proto_cpp(tensorflow/core/framework/attr_value.proto)
tf_proto_cpp(tensorflow/core/framework/tensor.proto)
tf_proto_cpp(tensorflow/core/framework/resource_handle.proto)
tf_proto_cpp(tensorflow/core/framework/tensor_shape.proto)
tf_proto_cpp(tensorflow/core/framework/types.proto)
tf_proto_cpp(tensorflow/core/framework/node_def.proto)
tf_proto_cpp(tensorflow/core/framework/full_type.proto)
tf_proto_cpp(tensorflow/core/framework/op_def.proto)
tf_proto_cpp(tensorflow/core/framework/versions.proto)
tf_proto_cpp(tensorflow/core/framework/kernel_def.proto)
tf_proto_cpp(tensorflow/core/grappler/costs/op_performance_data.proto)
tf_proto_cpp(tensorflow/core/protobuf/device_properties.proto)

# A python interpreter is required to produce the plugin wheel. This python environment
# must have the 'wheel' package installed.
find_package(Python 3.6 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "import wheel"
    RESULT_VARIABLE python_wheel_check_exit_code
    OUTPUT_QUIET
)
if(NOT ${python_wheel_check_exit_code} EQUAL 0)
    message(
        FATAL_ERROR 
        "The python interpreter at '${Python_EXECUTABLE}' does not "
        "have the 'wheel' package installed."
    )
endif()

set(CMAKE_FOLDER "")