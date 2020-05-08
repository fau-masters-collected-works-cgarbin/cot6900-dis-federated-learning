Federate Learning, a guided tour

*Learn from everyone, without learning about any one* [credit](https://federated.withgoogle.com/)

# Federated learning in under one hour

1. [Google's Federated Learning
  comics](https://federated.withgoogle.com/) explains the
  problem that federated learning solves and how it solves it. It
  also has references to other articles to learn more.
1. "[Are you a
  dog?](https://security.googleblog.com/2014/10/learning-statistics-with-privacy-aided.html)"
  Explains how to collect (aggregated) data privately (why
  "aggregated" is important). This is a fundamental concept of
  federated learning.

# Seminal works

Where federated learning came from, seminal papers and articles

1. [An article in Google's AI
  blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
  from a subset of the authors of the seminal paper (next bullet),
  explaining the federated learning concepts at a high level. It's
  recommended to read this blog post first, then the seminal paper
  (next bullet).
1. The first paper (2016) proposing the term "federated learning" is
  from a Google team working mobile devices,
  ["Communication-Efficient Learning of Deep
  Networks from Decentralized
  Data"](https://arxiv.org/abs/1602.05629): "*We advocate an
  alternative that leaves the training data distributed on the
  mobile devices, and learns a shared model by aggregating
  locally-computed updates. We term this decentralized approach
  **Federated Learning.** We present a practical method for the
  federated learning of deep networks based on iterative model
  averaging, and conduct an extensive empirical evaluation,
  considering five different model architectures and four
  datasets.*"
1. [Wikipedia's Federated Learning
  entry](https://en.wikipedia.org/wiki/Federated_learning)
  credits two earlier papers as "*...the first publications on
  federative averaging in telecommunication settings.*" The first is
  "*[Federated Optimization: Distributed
  Optimization Beyond the
  Datacenter](https://arxiv.org/pdf/1511.03575.pdf)"* and the
  second is "[*Federated Optimization:
  Distributed Machine Learning for On-Device
  Intelligence*](https://arxiv.org/abs/1610.02527)". Both are
  from the same set of three authors. Two of the authors are also
  coauthors of Google's 2016 federated learning paper.
1. "[Towards Federated Learning at Scale:
  System Design](https://arxiv.org/abs/1902.01046)", a Google
  paper from 2019, with some of the same authors from the papers
  above, addresses the practical aspects of deploying federated at
  scale. While the authors make it clear early on that not all
  problems are solved, they address "*...numerous practical issues:
  device availability that correlates with the local data
  distribution in complex ways (e.g., time zone dependency);
  unreliable device connectivity and interrupted execution;
  orchestration of lock-step execution across devices with varying
  availability; and limited device storage and compute resources.
  These issues are addressed at the communication protocol, device,
  and server levels. We have reached a state of maturity sufficient
  to deploy the system in production and solve applied learning
  problems over tens of millions of real-world devices; we
  anticipate uses where the number of devices reaches billions.*"

# Where federated learning is today

"[Federated Learning in Mobile Edge Networks: A Comprehensive
  Survey](https://arxiv.org/abs/1909.11875)", a recent paper,
  submitted in September of 2019 and revised in February of 2020. It
  has a well-written summary of how federated learning works, its
  applications, and challenges.

# Try it out

1. [Google's tutorial for image classification with federated learning](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)
