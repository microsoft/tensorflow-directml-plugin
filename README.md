# TensorFlow-DirectML-Plugin <!-- omit in toc -->

[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning. This repository is an implementation of [TensorFlow's Pluggable Device API](https://blog.tensorflow.org/2021/06/pluggabledevice-device-plugins-for-TensorFlow.html) that leverages [DirectML](https://github.com/microsoft/DirectML) to provide cross-vendor hardware acceleration on Windows 10 and the Windows Subsystem for Linux (WSL). TensorFlow with DirectML enables training and inference of complex machine learning models on a wide range of DirectX 12-compatible hardware.

## Questions, Issues, and Feedback

- Frequently asked questions: [FAQ](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-faq)
- Ask a question: [Discussions](https://github.com/microsoft/tensorflow-directml-plugin/discussions)
- Report a bug: [Issues](https://github.com/microsoft/tensorflow-directml-plugin/issues) 

You can also contact us directly at askdirectml@microsoft.com.

## Getting Started

TensorFlow DirectML Plugin is in early development and is not supported for production yet. For production scenarios, use [TensorFlow 1.15 with DirectML](https://github.com/microsoft/tensorflow-directml) instead.

The following resources provide additional background on DirectML and TensorFlow:
- [DirectML GitHub](https://github.com/microsoft/DirectML/)
- [RFC: TensorFlow on DirectML](https://github.com/tensorflow/community/pull/243)
- [TensorFlow homepage](https://www.tensorflow.org/)

## System Requirements

### Windows 10

* Windows 10 Version 1709, 64-bit (Build 16299 or higher)
* Python x86-64 3.7, 3.8 or 3.9<sup>1</sup>
* One of the following supported GPUs:
  * AMD Radeon R5/R7/R9 2xx series or newer
  * Intel HD Graphics 5xx or newer
  * NVIDIA GeForce GTX 9xx series GPU or newer

### Windows Subsystem for Linux

* Windows 10 Insider Preview, 64-bit (Build 20150 or higher)
* Python x86-64 3.7, 3.8 or 3.9<sup>2</sup>
* One of the following supported GPUs:
  * AMD Radeon R5/R7/R9 2xx series or newer, and [20.20.01.05 WSL driver](https://www.amd.com/en/support/kb/release-notes/rn-rad-win-wsl-support)
  * Intel HD Graphics 6xx or newer, and [28.20.100.8322 WSL driver](https://downloadcenter.intel.com/download/29526)
  * NVIDIA GeForce GTX 9xx series GPU or newer, and [460.20 WSL driver](https://developer.nvidia.com/cuda/wsl/download)

## Contribute

If you would like to contribute to tensorflow-directml-plugin, please see our [contribution guidelines](CONTRIBUTING.md) and read the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct). We use [GitHub issues](https://github.com/microsoft/tensorflow-directml-plugin/issues) for tracking requests and bugs. **Please do not report security vulnerabilities through public GitHub issues.** See SECURITY.md for more details.

See [BUILD.md](BUILD.md) for instructions on how to produce private builds of tensorflow-directml-plugin.

## License

This project is licensed under [MIT License](LICENSE).

Some files and code snippets originate from the TensorFlow repository and are licensed under the Apache License 2.0.

The tensorflow-directml-plugin Python wheel binary package includes a redistributable version of the DirectML library, which is downloaded automatically as a part of the build. The use of the redistributable DirectML library is governed by a separate license that is found as part of the package (found in `tensorflow-plugins/directml/DirectML_LICENSE.txt` when extracted).

## Data Collection Notice

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

### Disabling Telemetry

The official builds of tensorflow-directml-plugin (hosted on [PyPI](https://pypi.org/project/tensorflow-directml-plugin/)) have data collection enabled. This telemetry is enabled when building with `--config=dml_telemetry` (i.e. the `--telemetry` switch in `build.py`), but it is disabled by default for local builds.

## Trademarks Notice

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

**TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.**