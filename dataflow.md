# Data Flow (Under Construction)

The exploding popularity of deep learning has sparked of wave of domain-specific architectures to accelerate deep neural networks.
Reducing training times, scaling model sizes, and improving energy efficiency are now the priorities for hardware designers.
GPUs started the deep learning revolution by demonstrating the feasibility of training large networks using off-the-shelf hardware.
However, existing throughput-centric architectures like GPUs suffer from many of the same inefficiencies as CPUs; also, system architectures with limited memory capacity or interconnect bandwidth cap the achievable performance on these workloads.

**The importance of flexible hardware:** Flexibility is the key requirement for future data analytics accelerators.
These accelerators must support more than tensor operations since data-centric AI requires auxiliary data pre-processing, cleaning, and augmentation.
Data center operators avoid fixed-function hardware to simplify management and avoid over-provisioning.
Hardware must also last multi-year deployments and be able to accommodate constantly changing applications in a field with rapidly advancing algorithms and models.
To succeed, future architectures must be multi-purpose since large non-recurring engineering (NRE) costs prohibits fixed-function designs.

Furthermore, data analytics pipelines now incorporate pre-processing steps that differ significantly from the dense matrix kernels at the core of training.
Models operate on varied data types too, including structured, unstructured, or semi-structured data---text, tabular data, video, images, audio, etc.
As datasets grow and new hardware reduces training times, ETL (extract, transform, and load) pre-processing times can dominate runtime.
Sparse models and graph processing also necessitate architectural support for sparse matrix operations.
Architectures that only excel at matrix operations lack the flexibility to execute these pipelines end-to-end and must offload them to the host.


**Dataflow architectures:** There is always a tension between generality and performance in hardware architectures.
CPUs excel at extracting implicit parallelism from irregular applications and provide good single-thread performance; however, they lack the floating point compute density and throughput needed for data analytics workloads.
Similarly, FPGAs are flexible but quickly become resource-constrained when synthesizing structures from soft logic.

Coarse-grained reconfigurable architectures approach the energy efficiency and performance of an ASIC while retaining the reconfigurability of an FPGA.
Furthermore, domain-specific languages for reconfigurable architectures raise the level of abstraction and improve developer productivity by avoiding the need to program these architectures with low-level RTL languages.
By providing common data movement and communication patterns that appear in many applications, these architectures sidestep the programmability pitfall of ASICs.

Reconfigurable dataflow accelerators (RDA) are architectures that map applications spatially across the chip to naturally exploit the abundant pipeline parallelism in these workloads.
As data flows through RDAs' deep pipelines, the natural spatial locality enables storing data in high-bandwidth, distributed on-chip memories without the need to materialize to a bandwidth-limited, global memory.
Consequently, these architectures can achieve very high compute utilization by keeping huge arrays of floating point units fed with many terabytes per second of on-chip memory bandwidth.

**Plasticine:** Plasticine ([Prabhakar et al](https://dl.acm.org/doi/10.1145/3140659.3080256)) is a reconfigurable dataflow architecture for data analytics.
Plasticine's fabric is a homogeneous grid of compute and memory tiles.
The compute tile is a 16-lane vector datapath that pipelines computations over six reconfigurable stages.
The memory tile is a programmer-managed scratchpad that provides access patterns needed to pipeline hierarchical loop nests with concurrent access for high-throughput.
Plasticine showed that by identifying a common set of general-purpose communication, compute, and data-movement patterns across many application domains, a single dataflow architecture can target multiple application domains without sacrificing performance.

Follow up work demonstrated that with small micro-architectural extensions, Plasticine can also accelerate sparse matrix and graph applications as well as SQL.
Unified data analytics accelerators avoid the communication bottleneck of transferring data over a bandwidth limited interconnect.
They also open the door to advances in training by enabling pre-processing and data augmentation to run in the inner loop without degrading performance.

* [Plasticine](https://ieeexplore.ieee.org/document/8192487) ISCA 2017: A reconfigurable architecture for parallel patterns
* [Spatial](https://dl.acm.org/doi/10.1145/3192366.3192379) PLDI 2018: A language and compiler for application accelerators
* [Gorgon](https://dl.acm.org/doi/10.1109/ISCA45697.2020.00035) ISCA 2020: A unified data analytics accelerator for in-database machine learning
* [Capstan](https://arxiv.org/abs/2104.12760) MICRO 2021: A reconfigurable dataflow accelerator for sparse matrix algebra and graphs
