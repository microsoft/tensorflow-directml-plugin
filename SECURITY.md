# Using TensorFlow Securely

This document discusses how to safely deal with untrusted programs (models or
model parameters), and input data. Below, we also provide guidelines on how to
report vulnerabilities in TensorFlow.

## TensorFlow models are programs

TensorFlow's runtime system interprets and executes programs. What machine
learning practitioners term
[**models**](https://developers.google.com/machine-learning/glossary/#model) are
expressed as programs that TensorFlow executes.  TensorFlow programs are encoded
as computation
[**graphs**](https://developers.google.com/machine-learning/glossary/#graph).
The model's parameters are often stored separately in **checkpoints**.

At runtime, TensorFlow executes the computation graph using the parameters
provided. Note that the behavior of the computation graph may change
depending on the parameters provided. TensorFlow itself is not a sandbox. When
executing the computation graph, TensorFlow may read and write files, send and
receive data over the network, and even spawn additional processes. All these
tasks are performed with the permissions of the TensorFlow process. Allowing
for this flexibility makes for a powerful machine learning platform,
but it has implications for security.

The computation graph may also accept **inputs**. Those inputs are the
data you supply to TensorFlow to train a model, or to use a model to run
inference on the data.

**TensorFlow models are programs, and need to be treated as such from a security
perspective.**

## Running untrusted models

As a general rule: **Always** execute untrusted models inside a sandbox (e.g.,
[nsjail](https://github.com/google/nsjail)).

There are several ways in which a model could become untrusted. Obviously, if an
untrusted party supplies TensorFlow kernels, arbitrary code may be executed.
The same is true if the untrusted party provides Python code, such as the
Python code that generates TensorFlow graphs.

Even if the untrusted party only supplies the serialized computation
graph (in form of a `GraphDef`, `SavedModel`, or equivalent on-disk format), the
set of computation primitives available to TensorFlow is powerful enough that
you should assume that the TensorFlow process effectively executes arbitrary
code. One common solution is to whitelist only a few safe Ops. While this is
possible in theory, we still recommend you sandbox the execution.

It depends on the computation graph whether a user provided checkpoint is safe.
It is easily possible to create computation graphs in which malicious
checkpoints can trigger unsafe behavior. For example, consider a graph that
contains a `tf.cond` depending on the value of a `tf.Variable`. One branch of
the `tf.cond` is harmless, but the other is unsafe. Since the `tf.Variable` is
stored in the checkpoint, whoever provides the checkpoint now has the ability to
trigger unsafe behavior, even though the graph is not under their control.

In other words, graphs can contain vulnerabilities of their own. To allow users
to provide checkpoints to a model you run on their behalf (e.g., in order to
compare model quality for a fixed model architecture), you must carefully audit
your model, and we recommend you run the TensorFlow process in a sandbox.

## Accepting untrusted Inputs

It is possible to write models that are secure in a sense that they can safely
process untrusted inputs assuming there are no bugs. There are two main reasons
to not rely on this: first, it is easy to write models which must not be exposed
to untrusted inputs, and second, there are bugs in any software system of
sufficient complexity. Letting users control inputs could allow them to trigger
bugs either in TensorFlow or in dependent libraries.

In general, it is good practice to isolate parts of any system which is exposed
to untrusted (e.g., user-provided) inputs in a sandbox.

A useful analogy to how any TensorFlow graph is executed is any interpreted
programming language, such as Python. While it is possible to write secure
Python code which can be exposed to user supplied inputs (by, e.g., carefully
quoting and sanitizing input strings, size-checking input blobs, etc.), it is
very easy to write Python programs which are insecure. Even secure Python code
could be rendered insecure by a bug in the Python interpreter, or in a bug in a
Python library used (e.g.,
[this one](https://www.cvedetails.com/cve/CVE-2017-12852/)).

### What is a vulnerability?

Given TensorFlow's flexibility, it is possible to specify computation graphs
which exhibit unexpected or unwanted behavior. The fact that TensorFlow models
can perform arbitrary computations means that they may read and write files,
communicate via the network, produce deadlocks and infinite loops, or run out
of memory. It is only when these behaviors are outside the specifications of the
operations involved that such behavior is a vulnerability.

A `FileWriter` writing a file is not unexpected behavior and therefore is not a
vulnerability in TensorFlow. A `MatMul` allowing arbitrary binary code execution
**is** a vulnerability.

This is more subtle from a system perspective. For example, it is easy to cause
a TensorFlow process to try to allocate more memory than available by specifying
a computation graph containing an ill-considered `tf.tile` operation. TensorFlow
should exit cleanly in this case (it would raise an exception in Python, or
return an error `Status` in C++). However, if the surrounding system is not
expecting the possibility, such behavior could be used in a denial of service
attack (or worse). Because TensorFlow behaves correctly, this is not a
vulnerability in TensorFlow (although it would be a vulnerability of this
hypothetical system).

As a general rule, it is incorrect behavior for Tensorflow to access memory it
does not own, or to terminate in an unclean way. Bugs in TensorFlow that lead to
such behaviors constitute a vulnerability.

One of the most critical parts of any system is input handling. If malicious
input can trigger side effects or incorrect behavior, this is a bug, and likely
a vulnerability.

## Vulnerabilities in TensorFlow-DirectML-Plugin or the DirectML Backend

If you believe you have found a security vulnerability in any of the source code related to TensorFlow-DirectML-Plugin or the DirectML backend that meets [Microsoft's definition of a security vulnerability](https://docs.microsoft.com/en-us/previous-versions/tn-archive/cc751383(v=technet.10)), please report it to the Microsoft Security Response Center (MSRC) at [https://msrc.microsoft.com/create-report](https://msrc.microsoft.com/create-report).

If you prefer to submit without logging in, send email to [secure@microsoft.com](mailto:secure@microsoft.com). If possible, encrypt your message with our PGP key; please download it from the the [Microsoft Security Response Center PGP Key page](https://www.microsoft.com/en-us/msrc/pgp-key-msrc).

You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Additional information can be found at [microsoft.com/msrc](https://www.microsoft.com/msrc).

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

  * Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
  * Full paths of source file(s) related to the manifestation of the issue
  * The location of the affected source code (tag/branch/commit or direct URL)
  * Any special configuration required to reproduce the issue
  * Step-by-step instructions to reproduce the issue
  * Proof-of-concept or exploit code (if possible)
  * Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

If you are reporting for a bug bounty, more complete reports can contribute to a higher bounty award. Please visit our [Microsoft Bug Bounty Program](https://microsoft.com/msrc/bounty) page for more details about our active programs.

We prefer all communications to be in English.

Microsoft follows the principle of [Coordinated Vulnerability Disclosure](https://www.microsoft.com/en-us/msrc/cvd).