**Federated learning**
**Summary of selected papers and articles**

# Introduction

This question summarizes federated learning papers reviewed for this class.

# Selected papers and articles

The following papers and articles were selected as reading material to
help answer the research questions.

(1) "Federated Learning - Building better products with on-device data
and privacy by default - An online comic from Google AI",
[https://federated.withgoogle.com/](https://federated.withgoogle.com/),
accessed 2020-03-20

> **Why it was selected**: It explains in a concise manner the privacy
> problem that federated learning resolves and how it resolves it in a
> large, distributed environment. It manages to keep the explanations at
> a high level, going into enough technical details, while keeping it
> understandable for readers without a background in machine learning.
>
> This is a good introduction to the motivation and implementation
> challenges of federated learning.

(2) "Learning statistics with privacy, aided by the flip of a coin",
[https://security.googleblog.com/2014/10/learning-statistics-with-privacy-aided.html](https://security.googleblog.com/2014/10/learning-statistics-with-privacy-aided.html),
accessed 2020-03-20

> **Why it was selected**: Google's "federated learning comic" page
> (55th.. panel) refers to this article as one of the sources for
> privacy-preserving analytics. It explains the process of collecting
> data without exposing identifiable user data, a key concept in the
> claim that federated learning preserves privacy.

(3) "Communication-Efficient Learning of Deep Networks from
Decentralized Data",
[https://arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629),
accessed 2020-03-20

> **Why it was selected**: Google's "federated learning comic" page
> (55th.. panel) claims that "\[t\]he federated learning approach for
> training deep networks was first articulated \[...\]" in this paper.
> The paper names the approach "federated learning" for the first time
> in a publication ("We term this decentralized approach Federated
> Learning.").
>
> Besides its historical importance, the paper focuses on the practical
> application of federated learning: "We present a practical method for
> the federated earning of deep networks based on iterative model
> averaging, and conduct an extensive empirical evaluation, considering
> five different model architectures and four datasets. These
> experiments demonstrate the approach is robust to the unbalanced and
> non-IID data distributions that are a defining characteristic of this
> setting. Communication costs are the principal constraint, and we show
> a reduction in required communication rounds by 10–100× as compared to
> synchronized stochastic gradient descent."
>
> This focus on the practical aspects is promising for the research
> questions outlined above.

(4) "Federated Learning: Collaborative Machine Learning without
Centralized Training Data",
[https://ai.googleblog.com/2017/04/federated-learning-collaborative.html](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
accessed 2020-03-20

> **Why it was selected**: It is a summary of the seminal federated
> learning paper ("Communication-Efficient Learning of Deep Networks
> from Decentralized Data") and also has pointers to other sources that
> may be useful for the research.

(5) "Federated learning",
[https://en.wikipedia.org/wiki/Federated\_learning](https://en.wikipedia.org/wiki/Federated_learning),
accessed 2020-03-20

> **Why it was selected**: Because federated learning is still a
> relatively new area, it is moving fast. Its Wikipedia page was
> selected as a form of "sanity test", to check that possibly relevant
> areas of the field are not being neglected during the research.

(6) "Federated Optimization: Distributed Optimization Beyond the
Datacenter",
[https://arxiv.org/abs/1511.03575](https://arxiv.org/abs/1511.03575),
accessed 2020-03-20

> **Why it was selected**: [Wikipedia's
> Federated Learning
> entry](https://en.wikipedia.org/wiki/Federated_learning)
> credits two earlier papers as "...the first publications on federative
> averaging in telecommunication settings." One of them is this one, the
> other is the next entry in this list. Both of them are from the same
> set of authors. Two of these authors are also coauthors of Google's
> 2016 seminal federated learning paper, "Communication-Efficient
> Learning of Deep Networks from Decentralized Data".
>
> This paper and the next paper were selected for the research because
> they seem to be building blocks for the federated learning approach
> articulated more fully later. As such, they may give insights on the
> early challenges and solutions.

(7) "Federated Optimization: Distributed Machine Learning for On-Device
Intelligence",
[https://arxiv.org/abs/1610.02527](https://arxiv.org/abs/1610.02527),
accessed 2020-03-20

> **Why it was selected**: please refer to the previous entry.

(8) "Towards Federated Learning at Scale: System Design",
[https://arxiv.org/abs/1902.01046](https://arxiv.org/abs/1902.01046),
accessed 2020-03-20

> **Why it was selected**: It is another paper that focuses on practical
> aspects "Our work addresses numerous practical issues: device
> availability that correlates with the local data distribution in
> complex ways (e.g., time zone dependency); unreliable device
> connectivity and interrupted execution; orchestration of lock-step
> execution across devices with varying availability; and limited device
> storage and compute resources. These issues are addressed at the
> communication protocol, device, and server levels. We have reached a
> state of maturity sufficient to deploy the system in production and
> solve applied learning problems over tens of millions of real-world
> devices; we anticipate uses where the number of devices reaches
> billions.".

(9) "Federated Learning in Mobile Edge Networks: A Comprehensive
Survey",
[https://arxiv.org/abs/1909.11875](https://arxiv.org/abs/1909.11875),
accessed 2020-03-31

> **Why it was selected**: It is a recent paper, submitted in September
> of 2019 and revised in February of 2020.

To have a sense of the timeline of the papers and articles, the
following table shows their original publication date (some of them have
been revised after those dates) and Google scholar citations, when
applicable. There is a cluster of authors, but it is more a reflection
of selection bias than anything else. It does, however, show that Google
has been active in this area, including practical applications, for a
number of years.

| | Item | Authors                                                                                                                   | First published date                                                                                                                                                                                                                     | Revised date    | Cited by\[1\] |     |
| - | ---- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ------------- | --- |
| 1    | Federated Learning - Building better products with on-device data and privacy by default - An online comic from Google AI |                                                                                                                                                                                                                                          | 2019-05-07\[2\] |               |     |
| 2    | Learning statistics with privacy, aided by the flip of a coin                                                             | Úlfar Erlingsson                                                                                                                                                                                                                         | 2014-10-30      |               |     |
| 3    | Communication-Efficient Learning of Deep Networks from Decentralized Data                                                 | H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas                                                                                                                                                      | 2016-02-17      | 2017-02-28    | 743 |
| 4    | Federated Learning: Collaborative Machine Learning without Centralized Training Data                                      | Brendan McMahan and Daniel Ramage, Research Scientists                                                                                                                                                                                   | 2017-04-06      |               |     |
| 5    | Wikipedia's federated learning page                                                                                       |                                                                                                                                                                                                                                          | 2019-06-08\[3\] | 2020-03-08    |     |
| 6    | Federated Optimization:Distributed Optimization Beyond the Datacenter                                                     | Jakub Konečný, Brendan McMahan, Daniel Ramage                                                                                                                                                                                            | 2015-11-11      |               | 115 |
| 7    | Federated Optimization: Distributed Machine Learning for On-Device Intelligence                                           | Jakub Konečný, H. Brendan McMahan, Daniel Ramage, Peter Richtárik                                                                                                                                                                        | 2016-10-08      |               | 195 |
| 8    | Towards Federated Learning at Scale: System Design                                                                        | Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander | 2019-02-04      | 2019-03-22    | 178 |
| 9    | Federated Learning in Mobile Edge Networks: A Comprehensive Survey                                                        | Wei Yang Bryan Lim, Nguyen Cong Luong, Dinh Thai Hoang, Yutao Jiao, Ying-Chang Liang, Qiang Yang, Dusit Niyato, Chunyan Miao                                                                                                             | 2019-09-26      | 2020-02-28    | 15  |

*Table 1 - Authors, publication data, revision date and Google scholar
citations for the papers.*

- \[1\]  From Google scholar
- \[2\]  From the [web archive](https://web.archive.org/web/20190201000000*/https://federated.withgoogle.com/)
- \[3\] From the page history


The following sections summarize each item in the order listed above.
This order builds from the basics to more advanced topics.

# (1) Federated Learning - Building better products with on-device data and privacy by default - An online comic from Google AI

  - Federated learning "can handle our privacy concerns and improve
    functionality".
  -  "It lets us do machine learning while keeping data on device",
     "resilient", "low-impact", "secure".
  -  "The real-world performance of your machine learning model depends
     on the relevance of the data used to train it - and the best data
     lives right at the source: on the devices we use every day."
  -  "...but what if the data never leaves the device?"
  -  **Federated learning → train a centralized model on decentralized
     data.**
  -  Training is brought to the device.
  -  Not all devices participate. Devices must be **eligible**, e.g.
     must be charging, using Wi-Fi, idle.
  -  Devices receive a **training model** of only a few megabytes
      -  But it doesn't explain how the model is so small
  -  The device trains the model (in "only a few minutes") and sends
     the **training results, not the data,** to the server.
  -  **Secure aggregation** prevents the server from reconstructing the
     data from the trained model. "On each device, before anything is
     sent, the secure aggregation protocol adds zero-sum masks to
     scramble the training results. When you add up all those training
     results the masks exactly cancel out\!"
      -  "**Secure aggregation** is an interactive cryptographic
         protocol for computing sums of masked vectors, like model
         weights. It works by coordinating the exchange of random masks
         among pairs of participating clients, such that the masks
         cancel out when a sufficient number of inputs are received. To
         read more about secure aggregation, see [Practical Secure
         Aggregation for Privacy-Preserving Machine
         Learning](https://ai.google/research/pubs/pub47246)."
  -  What if a device has unique data? This is called **model
     memorization** (of specific user data) and needs to be avoided.
     "This is why we need to have ways to measure and control how much
     a model might be memorizing." This leads to **differential
     privacy** (limits how much a device contributes and adds noise to
     obscure rare data).
      -  "Understanding and mitigating the risks of **model
         memorization** is an active area of research. Techniques to
         measure memorization are explored, e.g. in the 2018 paper [The
         Secret Sharer: Measuring Unintended Neural Network
         Memorization & Extracting
         Secrets](https://g.co/research/SecretSharerPaper).
         Memorization risk can be mitigated by pre-filtering rare or
         sensitive information before training. More sophisticated
         mitigation techniques include differentially private model
         training as explored, for example, in the 2018 paper [Learning
         Differentially Private Recurrent Language
         Models](https://g.co/research/DPLanguageModelsPaper), which
         shows how to learn model weights that are not too dependent on
         any one device’s data. For more information on differential
         privacy, the canonical textbook “The Algorithmic Foundations
         of Differential Privacy” by Cynthia Dwork and Aaron Roth is
         available from NOW publishers and
         [online](https://www.cis.upenn.edu/~aaroth/privacybook.html).
  -  Testing of the updated model is also done on the devices. Some
     devices are set aside for training and others for training.
  -  Once the improvement is measured and deemed sufficient enough, all
     devices get the new model.
  -  At this point the model is static. To change the model we repeat
     the training/testing sequence.
  -  The "utility vs. privacy conflict"
      -  "**Federated learning and analytics** come from a rich
         heritage of distributed optimization, machine learning and
         privacy research. They are inspired by many systems and tools,
         including [MapReduce](https://en.wikipedia.org/wiki/MapReduce)
         for distributed computation,
         [TensorFlow](https://www.tensorflow.org/) for machine learning
         and
         [RAPPOR](https://security.googleblog.com/2014/10/learning-statistics-with-privacy-aided.html)
         for privacy-preserving analytics. The federated learning
         approach for training deep networks was first articulated in a
         2016 paper published by Google AI researchers:
         [Communication-Efficient Learning of Deep Networks from
         Decentralized Data](https://arxiv.org/abs/1602.05629)."
  -  Other use cases for aggregated, yet private data:
      -  Self-driving cars
      -  Disease diagnosis
  -  "\[L\]earn from **everyone**, without learning about **any one**."

# (2) Learning statistics with privacy, aided by the flip of a coin

  -  This article is referred to in the "Federated Learning comics", as
     an inspiration for federated learning.
  -  RAPPOR - Randomized Aggregatable Privacy-Preserving Ordinal
     Response: "...way to learn software statistics that we can use to
     better safeguard our users’ security, nd bugs, and improve the
     overall user experience."
  -  Built on the concept of **randomized response**
      -  "Randomised response is a research method used in
         structured survey interview. It was first proposed by S. L.
         Warner in 1965\[1\] and later modified by B. G. Greenberg in
         1969.\[2\] It allows respondents to respond to sensitive
         issues (such as criminal behavior or sexuality) while
         maintaining confidentiality. Chance decides, unknown to the
         interviewer, whether the question is to be answered
         truthfully, or "yes", regardless of the truth."
         [https://en.wikipedia.org/wiki/Randomized\_response](https://en.wikipedia.org/wiki/Randomized_response)
  -  "RAPPOR builds on the above concept, allowing software to send
     reports that are effectively indistinguishable from the results of
     random coin flips and are free of any unique identifiers. However,
     by aggregating the reports we can learn the common statistics that
     are shared by many users."
  -  "The guarantees of differential privacy, which are widely accepted
     as being the strongest form of privacy, have almost never been
     used in practice despite intense research in academia. RAPPOR
     introduces a practical method to achieve those guarantees."
      -  The statement "almost never been used in practice" may have
         changed since this article was published, in 2014.

# (3) Communication-Efficient Learning of Deep Networks from Decentralized Data

Because this paper claims the definition of "federate learning", I refer
to it as the "**original paper**" in other places.

This paper is dense. It's hard to make a summary that doesn't include
most of what it covers. The excerpts below cover key topics and
highlights then, in an attempt to make it easier to extract them from
the other good parts of the paper.

  -  The first definition of the term: "We advocate an alternative that
     leaves the training data distributed on the mobile devices, and
     learns a shared model by aggregating locally-computed updates.
     **We term this decentralized approach Federated Learning.**"
  -  This is a practical paper: "...and conduct an extensive empirical
     evaluation, considering five different model architectures and
     four datasets. These experiments demonstrate the approach is
     **robust to the unbalanced and non-IID data distributions** that
     are a defining characteristic of this setting. Communication costs
     are the principal constraint, and we show a reduction in required
     communication rounds by 10–100× as compared to synchronized
     stochastic gradient descent."
  -  Terminology: clients and server - "We term our approach Federated
     Learning, since the learning task is solved by a loose federation
     of participating devices (which we refer to as **clients**) which
     are coordinated by a central **server**."
  -  Surprised to see a reference to a government source here: "This is
     a direct application of the principle of focused collection or
     data minimization proposed by the 2012 White House report on
     privacy of consumer data \[39\]"
  -  "A principal advantage of this approach is the decoupling of model
     training from the need for direct access to the raw training
     data."
  -  "\[W\]e introduce the **FederatedAveraging algorithm**, which
     combines local stochastic gradient descent (SGD) on each client
     with a server that performs model averaging."
  -  "Ideal problems for federated learning have the following
     properties: 1) **Training on real-world data** from mobile devices
     provides a distinct advantage overtraining on proxy data that is
     generally available in the data center. 2) **This** **data is
     privacy sensitive or large in size** (compared to the size of the
     model), so it is preferable not to log it to the data center
     purely for the purpose of model training (in service of the
     focused collection principle). 3) For supervised tasks, **labels
     on the data can be inferred naturally from user interaction**."
  -  "\[T\]he source of the updates is not needed by the aggregation
     algorithm, so updates can be transmitted without identifying
     meta-data over a mix network such as Tor \[7\] or via a trusted
     third party."
  -  "Federated optimization has several key properties that
     differentiate it from a typical distributed optimization problem:
      -  Non-IID: Thetrainingdataonagivenclientistypically based on the
         usage of the mobile device by a particular user, and hence any
         particular user’s local dataset will not be representative of
         the population distribution.
      -  Unbalanced: Similarly, some users will make much heavier use
         of the service or app than others, leading to varying amounts
         of local training data.
      -  Massively distributed: We expect the number of clients
         participating in an optimization to be much larger than the
         average number of examples per client.
      -  Limited communication: Mobile devices are frequently offline
         or on slow or expensive connections."
  -  "In this work, our emphasis is on the non-IID and unbalanced
     properties of the optimization, as well as the critical nature of
     the communication constraints."
  -  "**\[W\]e use a controlled environment that is suitable for
     experiments, but still addresses the key issues of client
     availability and unbalanced and non-IID data.**"
  -  The process: "We assume a synchronous update scheme that proceeds
     in rounds of communication.". Steps (in a bullet list and somewhat
     edited from the actual text):
      -  There is a fixed set of K clients, each with a fixed local
         dataset.
      -  At the beginning of each round, a random fraction C of clients
         is selected
          -  We only select a fraction of clients for efficiency, as
             our experiments show diminishing returns for adding more
             clients beyond a certain point.
      -  \[T\]he server sends the current global algorithm state to
         each of these clients (e.g., the current model parameters).
      -  Each selected client then performs local computation based on
         the global state and its local dataset,
      -  \[Each client\] sends an update to the server. The server then
         applies these updates to its global state, and the process
         repeats.
  -  Communication cost vs. computation cost: "In data center
     optimization, communication costs are relatively small, and
     computational costs dominate, with much of the recent emphasis
     being on using GPUs to lower these costs. In contrast, in
     federated optimization communication costs dominate — we will
     typically be limited by an upload bandwidth of 1 MB/s or less.
     "... "On the other hand, since any single on-device dataset is
     small compared to the total dataset size, and modern smartphones
     have relatively fast processors (including GPUs), computation
     becomes essentially free compared to communication costs for many
     model types."... "\[T\]he speedups we achieve are due primarily to
     adding more computation on each client, once a minimum level of
     parallelism over clients is used."
  -  The **FederatedAveraging** **Algorithm** - this is a key concept
     of this paper
           -  **FederatedSGD** (FedSGD): "each client locally takes one step
         of gradient descent on the current model using its local data,
         and the server then takes a **weighted average** of the
         resulting models."
           -  **FederatedAveraging** (FedAvg): "Once the algorithm is
         written this way, we can add more
           -  computation to each client by iterating the local update
         multiple times before the averaging step."
           -  "The amount of computation is controlled by **three key
         parameters: C, the fraction of clients that perform
         computation on each round; E, the number of training passes
         each client makes over its local dataset on each round; and B,
         the local minibatch size used for the client updates**."
           -  Important problem with the weighted average: **if the local
         models start from different initial conditions, averaging them
         produces bad results in practice**. However, if we "start two
         models from the same random initialization and then again
         train each independently on a different subset of the data (as
         described above), we find that naive parameter averaging works
         surprisingly well".
           -  ![](media/image9.png)
       -  Experimental results with MNIST and a DNN:
           -  Goal: measure the number of communication rounds to achieve a
         predefined accuracy.
           -  Using about 10% of the clients (C=0.1) and batch size 10
         (B=10) significantly reduces the number of communication
         rounds. Based in this initial experiment, C was set to 0.1 for
         the other experiments in this section.
           -  With C set to C=0.1, increasing the number of computations
         performed in the clients (i.e. increasing the number of local
         SGD updates) by reducing B or increasing E (epochs)
         dramatically decreases communication rounds. It requires
         between 4x and 8x fewer rounds to achieve the desired
         accuracy, compared to C=0 (one client at a time, i.e. without
         federation).
           -  The improvement is noticeable even in a pathological non-IID
         case, where devices had only one of the MNIST digits to train
         on. In this case, the improvement is not as large as the IID
         case, but it is still significant.
           -  There is a risk of over-optimization. More local training
         resulted in smaller improvements or even divergence. To
         counteract that, later optimization cycles should increase B
         or decrease E to reduce the number of local computations (and
         thus decrease over-optimization).
           -  Experimental results with CIFAR-10 + CNN and text corpus +
         LTSM show a similar reduction in communication rounds.
       -  Future work: investigate stronger privacy guarantees with
     differential privacy or secure multi-party computation.

# (4) Federated Learning: Collaborative Machine Learning without Centralized Training Data

This article from Google's AI blogs is from two of the authors of the
paper above.

It expands on that paper by focusing on even more practical aspects of
an application in real-life conditions.

1.  It uses a real application (Google's
     [Gboard on
     Android](https://blog.google/products/search/gboard-now-on-android/))
     with user-input (as opposed to predetermined datasets).
2.  It deals with upload speeds being usually much slower than
     download speeds by [compressing
     updates](https://arxiv.org/abs/1610.05492) using random
     rotation and quantization (results in 100x smaller communication
     costs).
3.  It hides individual user data from the server with a
     [secure aggregation
     solution](https://eprint.iacr.org/2017/281).

The article also articulates better the benefits for the end-users:
"Federated Learning allows for smarter models, lower latency, and **less
power consumption**, all while **ensuring privacy**. And this approach
has another immediate benefit: in addition to providing an update to the
shared model, **the improved model on your phone can also be used
immediately**, powering experiences **personalized** by the way you use
your phone."

It narrow downs the scenarios that won't work with federated learning:
"Federated Learning **can't solve all machine learning problems** (for
example, learning to recognize different dog breeds by training on
carefully labeled examples), and for many other models the necessary
training data is already stored in the cloud (like training spam filters
for Gmail)."

Finally, it ends with tips to apply federated learning: "Applying
Federated Learning requires machine learning practitioners to adopt new
tools and a new way of thinking: **model development, training, and
evaluation with no direct access to or labeling of raw data, with
communication cost as a limiting factor.**"

# (5) Wikipedia's Federated learning

  -  A more general definition of federated learning, covering both the
     decentralization part, as well as the non-IID part (sometimes
     missed in the discussions): "\[A\] machine learning technique that
     trains an algorithm across multiple decentralized edge devices or
     servers holding local data samples, without exchanging their data
     samples. This approach stands in contrast to traditional
     centralized machine learning techniques where all data samples are
     uploaded to one server, as well as to more classical decentralized
     approaches which assume that local data samples are identically
     distributed."

  -  Hyper-parameters
      -  Network topology: central server vs. peer-to-peer.
      -  Federated learning parameters:
          -  T - number of rounds
          -  K - number of nodes (devices)
          -  C - fraction of nodes (devices) used
          -  B - local batch size
      -  Other parameters
          -  N - number of local training iteration before polling
             (this seems to be the same as E in the original paper,
             local number of epochs)
          -  η - local learning rate
  -  Summary of the original paper, showing it went from federated SGD
     to federated learning:
      -  "Federated Stochastic Gradient Descent (**FedSGD**) - Deep
         learning training mainly relies on variants of stochastic
         gradient descent, where gradients are computed on a random
         subset of the total dataset and then used to make one step of
         the gradient descent. Federated stochastic gradient
         descent\[6\] is the direct transposition of this algorithm to
         the federated setting, but by using a random fraction C of the
         nodes and using all the data on this node. The gradients are
         averaged by the server proportionally to the number of
         training samples on each node, and used to make a gradient
         descent step."
      -  "Federative averaging - Federative averaging (**FedAvg**)\[7\]
         is a generalization of FedSGD, which **allows local nodes to
         perform more than one batch update on local data and exchanges
         the updated weights rather than the gradients**. The rationale
         behind this generalization is that in FedSGD, if **all local
         nodes start from the same initialization**, averaging the
         gradients is strictly equivalent to averaging the weights
         themselves. Further, averaging tuned weights coming from the
         same initialization does not necessarily hurt the resulting
         averaged model's performance."
  -  Properties of federated learning:
      -  Privacy by design: data remains local, although this can be
         weakened with some specific attacks.
      -  Personalization: fine-tune a global mode with local data (a
         transfer learn technique of sorts).
      -  Legal compliance: complies with "data minimization" and GDPR
         guidelines, allowing institutions to cooperate without
         exchanging data.

# (6) Federated Optimization: Distributed Optimization Beyond the Datacenter

Coauthors of this paper also participated in the original federated
learning paper. This paper was written before the original paper.

It lays out some of the important assumptions of federated learning
(compared to distributed learning): "...**each \[device\] has only a
tiny fraction of data available totally**; in particular, we expect the
number of data points available locally to be much smaller than the
number of devices. Additionally, since different users generate data
with different patterns, we assume that **no device has a representative
sample of the overall distribution**."

These assumptions, of an uneven number of instances and non-IID
instances ("no device has a representative sample"), are fundamental
assumptions of federated learning. Solving those problems is the focus
of this paper. Furthermore, "In this work, **we are particularly
concerned with sparse data, where some features occur on only a small
subset of nodes or data points.** We show that the sparsity structure
can be used to develop an effective algorithm for federated
optimization."

The practical problems addressed are:

  -  Communication costs: "...communication cost is by far the largest
     bottleneck, as exhibited by a large amount of existing work."
  -  Unsuitable algorithms: "...many state-of-the-art optimization
     algorithms are inherently sequential, relying on a large number of
     very fast iterations."

The formal definition of the problem to be solved:

![](media/image4.png)

Where f(w) is the same function we are trying to minimize in a
traditional machine learning algorithm, i.e. it is the loss function:

![](media/image18.png)

The federated optimization algorithm is based on the SVRG algorithm
(from one of the authors, discussed in a cited paper).

![](media/image13.png)

The SVRG algorithm fails with sparse data. To make it work, it is
modified in these ways:

1.  > Make the step size *h* in line 8 inversely proportional to the
     size of the local data (number of samples on the device).
2.  > Scale the stochastic updates with a diagonal matrix
     *S<sub>k</sub>* to adjust for the representation of each feature
     in the local data (device), compared to the global representation
     of that feature.
3.  > When aggregating the data, A second diagonal matrix *A* adjusts
     the features that appear in only a few nodes, allowing for larger
     optimization steps in those cases.

These adjustments allow the distribution of the optimization tasks
without making assumptions about the data on the devices. In other
words, it does not assume the data is uniformly distributed.

The algorithm was tested on a problem that is unbalanced by definition:
"The dataset presented here was generated based on public posts on a
large social network. We randomly picked 10, 000 authors that have at
least 100 public posts in English, and try to predict whether a post
will receive at least one comment".

The dataset has 2,166,693 samples. The number of posts per author varies
from 75 to 9,000. Each author was distributed to one device. This makes
the training unbalanced and non-IID.

The green and red lines in the graph below demonstrate how much faster
the proposed algorithm converges, compared to other algorithms.

![](media/image12.png)

# (7) Federated Optimization: Distributed Machine Learning for On-Device Intelligence

This paper shares authors with Google's federated learning paper. It was
published a few months before that paper. It describes the communication
cost problem in more details.

The authors define **"federated optimization"** in this paper: "We
introduce a new and increasingly relevant setting for distributed
optimization in machine learning, where the data defining the
optimization are unevenly distributed over an extremely large number of
nodes. The goal is to train a high-quality centralized model. We refer
to this setting as *Federated Optimization*."

As in other papers, the goal is to minimize the communication cost: " In
this setting, communication efficiency is of the utmost importance and
minimizing the number of rounds of communication is the principal goal."

While the method helps improve privacy by not sending data to a central
node, the authors do not claim that it solves all privacy issues:
"Clearly, some trust of the server coordinating the training is still
required, and depending on the details of the model and algorithm, the
updates may still contain private information." Additional methods are
needed to improve privacy: "If additional privacy is needed,
randomization techniques from differential privacy can be used."

The paper claims that this application is the first practical
application of distributed non-IID learning: "The main purpose of this
work is initiate research into, and design a **first practical
implementation of federated optimization**. Our results suggest that
with suitable optimization algorithms, very little is lost by not having
an IID sample of the data available, and that even in the presence of a
large number of nodes, we can still achieve convergence in relatively
few rounds of communication."

Compared to Google's original paper, this paper is much more technical,
by a large margin. For example, section 2 reviews related works in
detail, starting with gradient descent and stochastic gradient descent,
then moving on to randomized algorithms ("...combine the benefits of
cheap iterations of SGD with fast convergence of GD.").

Section 2.3, named "Distribute Setting", reviews the literature for that
topic. Besides reviewing the algorithms, it proposes a method to measure
the efficiency of distribution optimization. The method starts by
defining the behavior in a single machine:

![](media/image15.png)

Then it extends the model for distributed environments:

![](media/image6.png)

With this framework in place, the paper frames the problem of **existing
stochastic algorithms** as "doing very large number
![](media/image7.png)of very fast ![](media/image11.png) iterations. As
a result, even relatively small *c* can cause the practical performance
of those algorithms drop down dramatically, because
![](media/image19.png)."

This problem with stochastic algorithms motivated the creation of
**distributed algorithms**. The paper reviews several distributed
algorithms and their shortcomings.

After building up the problem from the basics of gradient descent to the
current state of distributed algorithms, the paper moves on to a class
of algorithms it calls **communication-efficient algorithms**. Using its
method to measure the efficiency of such algorithms, it describes this
class of algorithms as "... one should design algorithms with high
![](media/image17.png), in order to make the cost of communcation
*(sic)* *c* negligible."

Finally, after reviewing the different classes of algorithms and their
weaknesses, the paper arrives at **federated optimization** algorithms.
It reviews two algorithms that served as bases for the work, Stochastic
Variance Reduced Gradient (SVRG) and Distributed Approximate Newton
(DANE). The combination of these two algorithms resulted in the
algorithm proposed in this paper.

The main challenges for the federated optimization algorithms are "the
number of data points available to a given node can differ greatly from
the average number of data points available to any single node.
Furthermore, this setting always comes with the data available locally
being clustered around a specific pattern, and thus not be- ing a
representative sample of the overall distribution we are trying to
learn."

To deal with these challenges, the federated optimization algorithm:

1.  Adjusts the local step size based on how much data is available in
     the node (device).
2.  Aggregates updates from nodes based on their dataset sizes.
3.  Scales the gradient updates by a matrix that compensates for
     different feature representations in the nodes (to deal with the
     non-IID distribution of samples across the nodes).
4.  Adjusts the aggregated updates based on the representation in each
     node (if a feature is present in only a few nodes, the updates
     from those nodes will have higher weight).

The final version of the algorithm is shown below. The authors named it
"federated stochastic variance reduced gradient" (FSVRG), for one the
base algorithm used, SVRG.

![](media/image1.png)

The experiment used a logistic regression model to predict if posts by
an author in a social media network would receive at least one comment.

Experiments show that FSVRG needs significantly fewer communication
rounds than other algorithms to converge, even when using a non-IID
distribution of the dataset across the nodes (88% of the features appear
in fewer than 1,000 nodes, out of 10,000 nodes). The figure below is
similar to the one from the previous paper.

![](media/image10.png)

The authors suggest the following items for future research and
usability improvements:

1.  Develop an asynchronous version of the algorithm, applying updates
     as soon as they arrive.
2.  Understand the convergence properties of the algorithm, to open
     more areas of research.
3.  Study the algorithm for non-convex objectives, e.g. neural
     networks (see next entry for an application that uses neural
     networks).
4.  Update the local models, not only the central mode, with local
     data (personalized models).

# (8) Towards Federated Learning at Scale: System Design

This is another paper by Google, with some of the authors of the
original paper. It lists fourteen authors, giving an idea of the breadth
of the paper.

This paper focuses on practical application: "We have built a scalable
production system for Federated Learning in the domain of mobile
devices, based on TensorFlow."

It does not attempt to solve the asynchronous problem of contemporary
federated learning applications: "\[W\]e chose to focus on support for
synchronous rounds while mitigating potential synchronization overhead
via several techniques we describe subsequently."

However, it does use a neural network, a topic that was left open in
previous papers (e.g. paper 7): "\[o\]ur system enables one to train a
deep neural network, using TensorFlow."

The high-level architecture of the experiment:

1.  A global model is maintained in the cloud.
2.  Android phones perform local updates, with local data.
3.  The phones send their updates to the global node.
4.  Secure aggregation is used to make updates from individual phones
     uninspectable.

The experiment addresses:

1.  Device availability correlated to complex factors that drive local
     data distribution, e.g. time zone dependencies (in this case, the
     application collects words typed by a user - users are not using
     devices at the same time across the globe).
2.  Unreliable device connectivity.
3.  Interrupted execution.
4.  Limited device storage and compute resources.

The paper claims to have resolved these issues with sufficient
confidence to deploy it in production with tens of millions of devices.

The solution starts with the creation of a protocol for the
participants. There are two participants in federated learning:

1.  A central, cloud-based server - the **FL server**.

2.  The distributed **devices** (Android phones, in this case.

The devices take the initiative to advertise that they are ready to run
an **FL task** (a specific computation, e.g. training with a specific
set of hyperparameters or evaluate a model locally) for a given **FL
population** (the learning problem). A **round** is the process of
selecting devices to participate in the FL task. The FL server tells the
selected devices what to do via an **FL plan** (a TensorFlow graph and
instructions to execute it). The devices get an **FL checkpoint** (the
current state of the model), perform the local computations with their
local dataset and send the update to the FL server. The server updates
its (global) state based on these updates.

The process is illustrated in the following figure.

![](media/image21.png)

The protocol has these phases (note its synchronicity by design):

1.  Selection: devices that meet the **eligibility** criteria (e.g.
     connected to power and in a Wi-Fi network) check-in with the
     server. The server selects a subset of these devices.
2.  Configuration: the server sends the plan and the checkpoint
     (current state).
3.  Reporting: the server waits for the results and aggregates them
     with federated averaging. If enough devices report back, the
     global model is updated.

The following section in the paper document the device and the server
architecture in detail. For brevity, those sections are not covered
here.

A serious operational challenge for this approach is its lack of control
of the version of TensorFlow running on the devices. Some devices may be
running months-old versions of TensorFlow. Since the FL task is defined
in terms of TensorFlow graphs and operations, it may specify a graph or
an operation that is not available in older TensorFlow versions. This
problem is solved by keeping an inventory of what version of TensorFlow
each device is running and by applying transformations to the FL task to
the older versions if needed.

The paper lists the following areas to be investigated in the future:

1.  Bias: the eligibility criteria (e.g. must be on Wi-Fi) and the
     compute resources needed to perform the FL tasks (e.g. need at
     least 2GB of RAM) introduce selection bias. The solution deals
     with this problem by performing A/B experiments with live data to
     evaluate the aggregated model. It has not detected significant
     bias so far, but the experiments are limited.
2.  Convergence time: the solution converges slower than a centralized
     ML solution (dataset and models in the same data center). One of
     the possible reasons is that the solution is using hundreds of
     devices in parallel, even though many more are available.
     Increasing parallelism is an area to be improved.
3.  Bandwidth: even though the solution already uses compression
     techniques to reduce bandwidth, the authors suggest that using
     quantization could reduce bandwidth usage further.
4.  Generalization of the work, a.k.a "federation computation":
     although the solution was presented in terms of a ML task, the
     method could be generalized to other distributed tasks.

# (9) Federated Learning in Mobile Edge Networks: A Comprehensive Survey

*This is a good paper to read as a summary of the problem to be solved
and how it is being solved, with references to other sources. It also
has a good description of how federated learning works at the
low-levels, better than what I have seen in other papers. If I had to
recommend only one paper to read, this would probably be it.*

As a side note, this paper says about the paper first selected for this
reading (the one replaced by this one): "For existing surveys on FL, the
authors in \[34\] place more emphasis on discussing the architecture and
categorization of different FL settings to be used for the varying
distributions of training data."

Although this paper has "mobile" in the title, the concepts discussed
are generic. Mobile networks are used as a motivation to introduce
federated learning: "Traditional cloud-based Machine Learning (ML)
approaches require the data to be centralized in a cloud server or data
center. However, this results in critical issues related to unacceptable
latency and communication inefficiency. To this end, **Mobile Edge
Computing (MEC) has been proposed to bring intelligence closer to the
edge, where data is produced**. However, conventional enabling
technologies for ML at mobile edge networks still require personal data
to be shared with external parties, e.g., edge servers. Recently, **in
light of increasingly stringent data privacy legislations and growing
privacy concerns, the concept of Federated Learning (FL) has been
introduced.**"

From that starting point, it opens a new topic to be addressed:
"However, in a large-scale and complex mobile edge network,
**heterogeneous devices with varying constraints are involved.** This
raises challenges of communication costs, resource allocation, and
privacy and security in the implementation of FL at scale."

It summarizes the advantages of federated learning as follows:

  -  *"Highly efficient use of network bandwidth*: Less information is
     required to be transmitted to the cloud. For example, instead of
     sending the raw data over for processing, participating devices
     only send the updated model parameters for aggregation. As a
     result, this significantly reduces costs of data communication and
     relieves the burden on backbone networks.
  -  *Privacy*: Following the above point, the raw data of users need
     not be sent to the cloud. Under the assumption that FL
     participants and servers are non-malicious, this enhances user
     privacy and reduces the probability of eavesdropping to a certain
     extent. In fact, with enhanced privacy, more users will be willing
     to take part in collaborative model training and so, better
     inference models can be built.
  -  *Low latency*: With FL, ML models can be consistently trained and
     updated. Meanwhile, in the MEC paradigm, real-time decisions,
     e.g., event detection \[22\], can be made locally at the edge
     nodes or end devices. Therefore, the latency is much lower than
     that when decisions are made in the cloud before transmitting them
     to the end devices. This is vital for time critical applications
     such as self-driving car systems in which the slightest delays can
     potentially be life threatening \[13\]."

The last bullet is an interesting twist on federated learning. Local
inference, running models on the edge devices is not new. However, the
claim here is that such models can now be updated more frequently,
making them more useful. In other words, without federated learning, the
local models would become outdated to the point of being useless.

It lists the following current applications of federated learning (not
exhaustive), with references to their papers:

  -  Improve next-word text prediction - this is Google's original
     paper.
  -  Develop predictive models for diagnosis in health AI.
  -  Foster collaboration among multiple hospitals.
  -  Foster collaboration across government agencies.

The paper identifies the following challenges for federated learning
implementation at scale:

1.  Communication costs are still high, even though we are no longer
     sending raw data. DNNs have millions of parameters, resulting in
     larger amounts of data to be exchanged.
2.  Devices participating in training, especially in large networks,
     are heterogeneous.
3.  Although privacy is improved, it is not guaranteed in the presence
     of malicious participants or servers. A malicious actor can infer
     information from parameter updates.

According to the authors, these challenges are not covered in other
review papers. To address them, this paper reviews other sources of
information that concentrate on those topics.

The following picture shows the areas covered in the paper.

![](media/image14.png)

The following table shows the papers reviewed and what topics they
cover.

![](media/image8.png)

## Background and fundamentals

A conceptual model for federated learning:

  -  Step 1 - Task initialization: the server pushes the current model
     and training hyperparameters to the devices.
  -  Step 2 - Local model training an update: the devices perform
     training with their local data and send the resulting set of model
     parameters (usually weights) to the server.
  -  Step 3 - Global model aggregation ad update: the server aggregates
     the local models and sends an updated global mode back to the
     devices.

![](media/image16.png)

The same steps, described as an algorithm:

![](media/image5.png)

Federated learning presents a statistical challenge to the model
training process. The distribution of samples in the devices cannot be
controlled by a centralized server. There is no guarantee that the
distribution will be IID across the devices. In fact, it is likely that
they will not be.

The paper that originally proposed the federated learning algorithm
(*FedAvg*) concluded that this did not significantly affect the accuracy
of the glocal model. Later papers found that it can affect accuracy,
lowering it by as much as 51%.

Proposed solutions include:

  -  Send a shared dataset to the devices, not only the global model.
     The devices train on the local dataset and on the shared dataset.
  -  Devices to send their data distribution (not the data itself) to a
     central server. Devices are then told to perform data augmentation
     in the under-represented classes. In addition, a mediator selects
     devices with data distributions that best contribute to the
     training process.
  -  The devices train on customized loss functions that account for
     the data imbalance.
  -  Devices share a set of base layers that are trained with *FedAvg*,
     then each participant trains a set of *personalization layers*
     using their own local data (a concept similar to transfer
     learning). This method is called FEDPER.

The paper describes the federated learning protocol from "Towards
federated learning at scale: System design", including the same figure
to illustrate. That paper was also reviewed in this summary. Please
refer to that section.

Even though data remains local, it is still possible to recover private
information from the models. To reduce the chances of leaking data
through the model, privacy is enhanced through:

  -  Secure aggregation: devices communicate with a trusted
     third-party. This trusted entity hides the devices' identity.
  -  Differential privacy: noise is added to the local updates in a way
     that, when results are aggregated, the overall model does not lose
     accuracy.

Frameworks that implement federated learning:

1.  TensorFlow Federated (TFF): adds a federated learning layer to
     TensorFlow.
2.  PySyft: adds federated learning to PyTorch.
3.  LEAF: a framework of datasets for federated learning benchmark.
     For example, it has split the MNIST dataset by writer of each
     character (i.e. each writer represents a user with a device),
     creating the Federated Extended MNIST, FEMNIST).

### Communication cost

Models with millions of parameters, while smaller than dataset, are
still large. Downloading and uploading the models incur high
communication costs. These costs can be reduced by:

1.  Performing as much computation as possible in the devices, taking
     advantage of the increasing computing power of edge devices. This
     is done by increasing the number of epochs or decreasing the batch
     size.
2.  Compress the models, with techniques already used in distributed
     learning, such as sparsification, quantization, and subsampling.
     This may introduce noise, though.
3.  "Importance-base updating", i.e. decide if it is worth uploading
     an updated model.

Model compression techniques:

1.  Structured updates: send a sparse matrix.
2.  Sketch updates: subsample the weight matrix or quantize it.
3.  Lossy compression
4.  Federated dropout: similar to local dropout, remove some
     connections in the network, reducing the model size.

The following table summarizes the compression techniques:

![](media/image3.png)

Of all compression techniques, quantization seems more promising: "On
the other hand, quantization with Kashin’s representation can achieve
the same performance as the baseline without compression while having
communication cost reduced by nearly 8 times when the model is quantized
to 4 bits. For federated dropout approaches, the results show that a
dropout rate of 25% of weight matrices of fully-connected layers (or
filters in the case of CNN) can achieve acceptable accuracy in most
cases while ensuring around 43% reduction in the size of models
communicated. However, if dropout rates are more aggressive, convergence
of the model can be slower."

## Resource allocation

### Participant selection

Participant selection must balance selecting as many diverse
participants as possible without affecting training time. It may result
in not selecting devices that are not willing to participate because
they don't have enough compute power (CPU performance), enough battery
(e.g. not plugged to an outlet), or in an expensive network connection
(metered cellular data connections). In other words, the training
process may suffer from selection bias.

Several algorithms have been recently proposed. Most of them have been
verified in smaller networks, smaller datasets, or both. It's too early
to state we have a solution to the problem because the documented
attempts to solve the problem were done as lab experiments that may not
be representative of real-life conditions.

### Adaptive aggregation

The algorithm proposed in Google's original paper, FedAvg, is
synchronous. All participants start at the same time, from the same base
model. The server aggregates the updates once all participants have
submitted their updates (with some error handling to account for dropped
participants).

Tests showed that letting participants join a running model update
sessions still works, but convergence takes longer. The FedSync
algorithm attempts to solve that by weighting the updates. Participants
that started longer ago, with an older version of the model, are
weighted less. However, the algorithm needs to be tuned to ensure
convergence and has, therefore, not being used in large settings yet.

Because of these limitations, synchronous federated learning remains the
most commonly used form.

### Summary and lessons learned for resource allocation

![](media/image2.png)

## Privacy and security issues

### Privacy issues

*Inferring information from models*: Extracting private information from
models predates federated learning. This issue applies to federated
learning as well. Although the data remains on the device, the
centralized model is subject to the same attacks that result in a
certain amount of data leakage, enough to compromise privacy.

*Differential privacy* can be used to reduce data leakage. It requires a
large number of participants, to render the added masks (usually
Gaussian noise) statistically insignificant.

*Selective parameter sharing*: Even with differential privacy, a
malicious server could extract private data. To counter that, a solution
has been proposed where participants share only some parameter updates.
This reduces the amount of (model) data exposed and thus reduces the
amount of data that can be compromised. This method has been shown to
work with relatively simple tasks (MNIST), but has not been yet tested
on larger classification tasks.

*Federated GANs* have been proposed as a way of not exposing the actual
training data. Instead of training an inference model, the participants
train a GAN, which then produces synthetic samples. These samples are
then used to train the inference model. With this approach, the actual
data is never exposed. On the other hand, it suffers from the
limitations of GANs.

### Security Issues

*Data poisoning*: the server does not have access to the data and to the
labels the devices are using. A device can easily poison the centralized
model, even when using a small number of *dirty-label* samples. This
problem gets worse if multiple participants train with dirty labels
(*sybil-based* data poisoning). A sybil-based attack can be mitigated in
a non-IID dataset by inspecting the updated parameters arriving from the
devices (*FoolsGold* strategy). In this environment, devices should have
their own particular patterns in the updates. A set of similar updates,
from different devices, indicates a sybil-based attack.

*Model poisoning attacks*: in this attack, participants directly poison
the model they send to the server for update (NB: this seems to be a
different federated learning method - update the model, not parameters).
It is more effective than data poisoning because data poisoning is
scaled based on the dataset and number of participants, while in this
attack the model is affected. Mitigations include first testing the
model accuracy against a baseline and comparing the model against models
received from other participants.

*Free-riding attacks*: participants may decide to free-ride by running
local updates on a very small local dataset, thus sparing its resources,
while still benefiting from all other participants. The only proposed so
far is blockchain-based, which brings its own set of issues.

Summary of attacks and mitigations:

> ![](media/image20.png)

## Applications of federated learning for mobile edge computing

This section considers the following applications of federated learning
for mobile edge network optimizations:

  -  *Cyberattack detection*: It has been demonstrated that distributed
     learning is an effective way to detect cyberattacks. However, it
     needs potentially sensitive data from the network nodes. Federated
     learning solves that problem by keeping the data in the network
     nodes.
  -  *Edge caching and computation offloading*: Edge caching reduces
     access time to popular files by caching them closer to the
     devices. It needs to learn which files to cache and which to
     evict. Federated learning allows the cache servers to learn that
     information without exposing private information. Computation
     offloading attempts to decide if a computation task should be done
     on the edge device or offload to a central server. To make that
     decision it needs to train models based on device states (e.g. CPU
     capabilities, energy levels, etc.). Such models can be trained
     without exposing the device states
  -  *Base station association*: Base station association's goal is to
     optimize the handover of users from one base station to another.
     It needs two sets of potentially private data: different base
     stations have information about the same user (e.g. to try to
     predict when to handover) and a user's device needs information
     from other users on a given base station to make a decision. In
     both cases, models can be trained with federated learning to not
     expose private data.
  -  *Vehicular networks*: Vehicular networks need to be trained on
     datasets that are distributed by definition. In different
     applications, such as improving automated driving with captured
     images and distribution of energy reserves to charging stations,
     data is available at the edges of the network. Federated learning
     is used to train centralized models without exposing private data
     from the edge devices (cars and charging stations).

## Challenges and future research directions

  -  Dropped participants: a large number of dropped participants may
     invalidate a training round.
  -  Privacy concerns: the existing solutions, such as differential
     privacy, affect model performance (reduce accuracy).
  -  Unlabeled data: all solutions analyzed so far assume labeled data.
  -  Interference among mobile devices: participant selection takes
     into consideration the device's capabilities and state, but don't
     consider the proximity of those devices. Choosing a large set of
     devices that share the same communication channel (same cell tower
     or network segment) may congest that channel during model updates.
  -  Communication security: besides the data and the model update, the
     communication channel itself may be attacked (e.g. jamming
     attacks).
  -  Asynchronous FL: most of the current applications use synchronous
     FL, but asynchronous FL models real-life situations better (fast
     vs. slow devices, joining training round already in progress,
     etc.). New algorithms are emerging for asynchronous FL.
