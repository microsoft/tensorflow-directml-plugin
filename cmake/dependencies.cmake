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
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests")

# TensorFlow python package
if(WIN32)
    # To update the package version to a nightly build, go to https://pypi.org/project/tf-nightly-intel/#files and copy the
    # URLs and SHA256 hashes for the cp37 versions. For stable builds, go to https://pypi.org/project/tensorflow-intel/#files instead.
    FetchContent_Declare(
        tensorflow_whl
        URL https://files.pythonhosted.org/packages/4c/57/3b37cf30bf1549e617f8328c10b1f3237bf50c9ac7be3de6fcb4c73a19ef/tensorflow_intel-2.12.0rc0-cp39-cp39-win_amd64.whl
        URL_HASH SHA256=4f6793e0d9238b2fd57f10f39d7b3ab38000e2fca3144c260f581665be35a971
    )

    FetchContent_Declare(
        tensorflow_framework
        URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.10.0.zip
        URL_HASH SHA256=4c5e6f9a7683583220716fecadea53ace233f31f59062e83585c4821f9968438
    )
else()
    # To update the package version to a nightly build, go to https://pypi.org/project/tf-nightly-cpu/#files and copy the
    # URLs and SHA256 hashes for the cp37 versions. For stable builds, go to https://pypi.org/project/tensorflow-cpu/#files instead.
    FetchContent_Declare(
        tensorflow_whl
        URL https://files.pythonhosted.org/packages/61/f7/5888bb138d8ae9c6400996c3f0ffc2d9034c5213d7c6aa6deea79bbfe2c3/tensorflow_cpu-2.12.0rc0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        URL_HASH SHA256=505921b47561f7b26fef8d1e5d781c4afd9aaba8945b02258d36037f70cdfbef
    )

    FetchContent_Declare(
        tensorflow_framework
        URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz
        URL_HASH SHA256=141631c8bcebb469ba5fa7bc8032f0324df2a7dea469f842318bafe0322850ab
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
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.9.1
    URL_HASH SHA256=cc2951e10d22ae41902e91afe7c2e4b2322d5ed49b3f396a05ef9e27c78aaeff
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

# GoogleTest Library
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    squeezenet_model
    URL https://github.com/oracle/graphpipe/raw/v1.0.0/docs/models/squeezenet.pb
    URL_HASH SHA256=5922a640a9e23978e9aef1bef16aaa89cc801bc3a30f4766a8c8fd4e1c6d81bc
    DOWNLOAD_NO_EXTRACT TRUE
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
    googletest
    squeezenet_model
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
# message("tensorflow_generated_protobuf_dir is \"${tensorflow_generated_protobuf_dir}\"")
set(tensorflow_include_dir ${tensorflow_whl_SOURCE_DIR}/tensorflow/include)
# set(tsl_include_dir ${tensorflow_whl_SOURCE_DIR}/tensorflow)
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

add_library(tensorflow_python_libs INTERFACE)
target_link_libraries(
    tensorflow_python_libs
    INTERFACE
    $<$<BOOL:${WIN32}>:${tensorflow_whl_SOURCE_DIR}/tensorflow/python/_pywrap_tensorflow_internal.lib>
    $<$<BOOL:${UNIX}>:${tensorflow_whl_SOURCE_DIR}/tensorflow/python/_pywrap_tensorflow_internal.so>
    $<$<BOOL:${UNIX}>:${tensorflow_whl_SOURCE_DIR}/tensorflow/libtensorflow_framework.so.2>
)

add_library(tensorflow_framework_libs INTERFACE)
target_link_libraries(
    tensorflow_framework_libs
    INTERFACE
    $<$<BOOL:${WIN32}>:${tensorflow_framework_SOURCE_DIR}/lib/tensorflow.lib>
    $<$<BOOL:${UNIX}>:${tensorflow_framework_SOURCE_DIR}/lib/libtensorflow.so>
    $<$<BOOL:${UNIX}>:${tensorflow_framework_SOURCE_DIR}/lib/libtensorflow_framework.so>
)

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
    # message("Proto stem is \"${proto_stem}\"")
    cmake_path(GET proto_path PARENT_PATH proto_parent_dir)
    # message("Proto parent dir is \"${proto_parent_dir}\"")
    # message("tensorflow_generated_protobuf_dir is \"${tensorflow_generated_protobuf_dir}\"")
    cmake_path(SET proto_generated_h ${tensorflow_generated_protobuf_dir}/${proto_parent_dir}/${proto_stem}.pb.h)
    # message("proto_generated_h is \"${proto_generated_h}\"")
    cmake_path(SET proto_generated_cc ${tensorflow_generated_protobuf_dir}/${proto_parent_dir}/${proto_stem}.pb.cc)
    message("tensorflow_include_dir is \"${tensorflow_include_dir}\"")
    message("proto_path is \"${proto_path}\"")

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
tf_proto_cpp(tensorflow/core/profiler/protobuf/xplane.proto)
tf_proto_cpp(tensorflow/tsl/profiler/protobuf/xplane.proto)


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