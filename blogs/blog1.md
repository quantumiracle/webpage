# Foundational Video World Model (2024.12 WIP)

A foundational video world model is crucial for advancements in robotics (e.g., [video language planning](https://video-language-planning.github.io/), [diffusion forcing](https://boyuan.space/diffusion-forcing/)) and the next generation of game engines (e.g., [GameNGen](https://gamengen.github.io/), [GameGen-x](https://gamegen-x.github.io/), [Oasis](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model), [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)), autonomous driving, etc. In robotics, such a model could enable access to an infinite amount of interaction data within realistic environments, moving beyond the limitations of non-realistic simulators. This breakthrough has the potential to bypass the long-standing simulation-to-reality gap that has impeded the research community for over a decade.

<img src="/home/quantumiracle/research/webpage/blogs/files/vlp.gif" alt="vlp" style="zoom:400%;" />

​					<figcaption>Video source: video-language-planning project page.</figcaption>

![oasis](/home/quantumiracle/research/webpage/blogs/files/oasis.gif)

​									<figcaption>Video source: open-oasis model.</figcaption>

![gamegenx](/home/quantumiracle/research/webpage/blogs/files/gamegenx.gif)

​					        <figcaption>Video source: GameGen-x project page.</figcaption>

[OpenAI's blog on Sora](https://openai.com/index/video-generation-models-as-world-simulators/) project claims video model as the world simulators. While this concept is undoubtedly inspiring and promising, it is important to approach such claims with skepticism. This blog critically examines the desiderata for constructing foundational video models that are both practical and feasible for robotics and the game industry. Drawing from the current state of video generation techniques, the limitations of existing models are highlighted. To guide the research community towards meaningful progress, I have identified examples for each issue as evidence, accompanied by further analysis to illuminate potential research directions.

## Desiderata
To establish a foundational video world model, the following desiderata must be addressed:

### Fidelity and Realism

The model must accurately reflect physical laws for robotics or simulated rules in game engines. This fidelity ensures real-world applicability and effective simulation.

### Efficient Video Generation

The model should support efficient video generation with minimal computational cost and latency. Techniques such as few-step sampling, distillation, and hardware acceleration are essential to achieve faster and more resource-efficient results.

### Long Video Generation with Frame Conditioning

Long video sequences should be generated conditionally, leveraging previous or current frames as context. Autoregressive techniques are crucial to maintaining coherence over extended timelines.

### Historical Information Retrieval

The ability to utilize historical frames or retrieval-augmented generation (RAG) is critical for context-aware generation. Alternative memory mechanisms can help maintain computational efficiency while effectively integrating past information.

### Action Conditioning

The model should support agent interaction by processing action inputs and predicting future outcomes in video format. This capability is fundamental for applications in robotics and gaming.

### Unified Latent Action Representation

The model must accommodate diverse environments and agents. For games, this involves mapping hardware inputs for standard user controls; for robotics, both high-level abstract actions and precise low-level robot control commands should be seamlessly integrated.

### Controllability

A highly controllable model is essential for practical applications. Controllability allows for:

1. Reducing prediction uncertainty with external guidance.
2. Generating diverse samples under varying conditions.

Techniques such as classifier-free guidance and ControlNet enhancements offer promising ways to improve model controllability.

### 3D Information

While incorporating 3D information can enhance realism, it may not always be necessary. Robust 3D reconstruction from 2D videos or binocular vision can serve as an efficient alternative. The depth information is provided in some cameras. If 3D information can be reconstructed with 2D videos or binocular vision, we may not necessarily require the depth information for foundational video world models.

## Problems

### Partial Observability

Partial observability has long been a challenge in reinforcement learning, as most practical environments provide agents with only incomplete information. This inherent limitation creates theoretical difficulties for agents in optimizing their policies, even with their best estimates of belief states.

Similarly, when a video generation model functions as a world simulator, it also faces the partial observability problem. To accurately reconstruct the dynamics and motions of agents and objects within an environment, the model must intake sufficient environmental information. Without such input, it cannot reliably predict future trajectories.

Drawing an analogy to large language models (LLMs), this issue resembles the challenge of addressing vague or underspecified user queries. Current solutions involve multi-turn question-and-answer interactions between users and LLMs to refine problem specifications. A similar approach could potentially mitigate ambiguity for video generation models. However, facilitating effective communication between the video generation model and the agent using it is considerably more complex.

Another method to address partial observability is by concatenating more historical frames to provide additional conditioning information for the video generation model. However, this approach imposes a heavier computational burden on the model, as the increased conditioning information will very likely add to the complexity of attention computations during subsequent frame generation.

### Recency bias

<img src="/home/quantumiracle/research/webpage/blogs/files/mc_example.gif" alt="mc_example" style="zoom:150%;" />

​								 <figcaption>Video generated with open-oasis model.</figcaption>

The recency bias is related to the lack of long-term memory in the design of current video generation models. The example is based on [open-oasis](https://github.com/etched-ai/open-oasis) project. The model is [diffusion transformer](https://arxiv.org/abs/2212.09748) with [diffusion forcing](https://arxiv.org/abs/2407.01392) technique to enable **autoregressive** rollout. The video diffusion model is conditioned on sequential action inputs. As shown in the example, the long video generation will collapse once the agent starts to stare at the ground, which causes the loss of information about surrounding objects and is no longer recovered. This is the recency bias of video world model due to lack of memory mechanism in video generation process. The partially observed recent historical frames do not encompass complete information about the environment, thus causing information loss in future frame prediction. [Loopy](https://arxiv.org/abs/2409.02634) project proposes to proportionally sampling (with recency weights) from all historical frames as additional condition for video generation, which alleviates the problem. But too many historical frames will increase the burden for diffusion modeling. Effective representation of historical information can be essential.

### Compounding error

![diffusionforcing](/home/quantumiracle/research/webpage/blogs/files/diffusionforcing.gif)

​                             <figcaption>Video source: Diffusion Forcing project page.</figcaption>

Different from text generation in large language models, video generation models suffer from the compounding error if applied in an autoregressive manner. By conditioning on previous frames, the frame-to-frame prediction error accumulates when iteratively sampling from a video model. The teacher forcing method in above comparison clearly suffers from a severe compounding error.

### Physical reality

![non_physical](/home/quantumiracle/research/webpage/blogs/files/non_physical.gif)

​                                <figcaption>Video source: OpenAI Sora model.</figcaption>

Since the initial public announcement of the Sora project, it has been widely acknowledged that current video generation methods using diffusion models cannot fully guarantee physical reality, as shown in above video example. To date, no principled solution has been established to address this limitation. In game simulation, this issue is less critical compared to robotics video modeling, where adherence to the laws governing our physical world is highly influential for robot planning performances. While game simulations primarily demand visual realism (not even physical reality), existing techniques still fall short of ensuring that no artificial or unnatural processes occur during the simulation.

On one hand, practical observations confirm that larger models trained on more realistic datasets tend to generate video samples with greater fidelity and fewer observable artifacts. It is reasonable to anticipate that this issue could become less pronounced when models are trained on datasets several orders of magnitude larger. A successful analogy can be drawn from the ability of large language models (LLMs) to master linguistic grammar.

On the other hand, compared to linguistic grammar, physical laws are significantly more complex and structured, with an almost zero tolerance for errors. This complexity makes the emergence of a model capable of fully grasping physical laws purely from video data seem somewhat improbable. For example, LLMs, despite their advances, still struggle to fully understand mathematics or even basic arithmetic. This highlights a dual challenge: the difficulty LLMs face with mathematics parallels the challenge of capturing physics in foundational video models.

recent progress on this problem

### Unnatural Motion

![hunyuan_cat](/home/quantumiracle/research/webpage/blogs/files/hunyuan_cat.gif)

​        <figcaption>A cat walks on the grass, realistic style, by HunyuanVideo model.</figcaption>

Existing video generation models struggle significantly with  multi-object scenarios and fast motion, particularly when similar visual patterns occur in close proximity within a small spatial region across  frames. Examples include the rapid movements of human fingers, animal  legs, and entire animal bodies (as illustrated in the example above).  The primary challenge arises from the diffusion model's reliance on a  denoising process. In this context, nearby pixels that contain random noises tend to exhibit similarities in the latent noised space, leading to ambiguity for the diffusion model. This ambiguity hinders its ability to effectively differentiate between closely related yet distinct denoising trajectories.

### Controllability

Controllability is a crucial aspect of practical world simulators, extending beyond the generation of outcomes based solely on training distributions or  deterministic realistic trajectories. Due to inherent limitations in  training data coverage or insufficient information, model predictions  should maintain a certain level of uncertainty. In this context, the  controllability of the model serves at least two essential purposes:

1. Uncertainty Reduction: With additional guidance  provided, a highly controllable model should be able to minimize  uncertainty in future predictions.
2. Sample Diversity: Given various forms of guidance, the controllable model should generate diverse samples.

From an application perspective, controllable models offer  significant advantages. In game simulation, they can generate futures  with different properties, such as object layouts or environmental  states, based on various provided conditions. For robotic simulation,  enhanced controllability leads to more precise trajectory predictions.  Furthermore, the ability to introduce various conditions in controllable models can serve a purpose similar to domain randomization,  facilitating the development of robust robotic policies.

The key to achieving controllability lies in enhancing the model's dependency on externally provided additional conditions for conditional modeling. Current techniques for controllable generation include  [classifier-free guidance](https://arxiv.org/abs/2207.12598) in pre-training diffusion models and  architectural designs like [ControlNet](https://arxiv.org/abs/2302.05543), which improve controllability  during the fine-tuning process. From the aspect of architectural design of the diffusion transformer ([DiT]((https://arxiv.org/abs/2212.09748))), the conditioning mechanism can be integrated through: (1) self-attention applied to a fully concatenated sequence of both the conditioning data and the video sequence, (2) cross-attention mechanisms that directly attend from the video sequence to the conditioning sequence, or (3) adaptive LayerNorm, which injects the conditional signals into the network.

These approaches represent significant strides towards creating  more versatile and adaptable world simulators. However, as the field  progresses, further research is needed to refine these methods and  explore new avenues for enhancing model controllability across diverse  applications.

### Diverse Control Space

To establish a foundational video world model, it is essential that the  model accommodates diverse control mechanisms. This challenge is evident in both robotics and gaming video models. In gaming engines, the action space varies significantly from one game to another due to differences  in control design and hardware interfaces. For instance, in the game *Street Fighter*, apart from standard motions, players can perform at least three distinct actions for kicks and three for punches, along with additional special moves and action  combinations. Conversely, other games may lack these specific actions or exhibit substantial variations in their implementation. Furthermore,  the unification of hardware interfaces presents another challenge, as  users may input commands through keyboards, mice, gamepads, or other  devices. 

<img src="/home/quantumiracle/research/webpage/blogs/files/sf_control_pad.png" alt="sf_control_pad" style="zoom:50%;" />

In robotics, the challenge of developing a unified control space  for foundational world models is exacerbated by the unique embodiments  of different robots, which result in varying degrees of freedom and  distinct action spaces. Each robot's design influences its control  mechanisms, leading to differences in latency and control modes, such as position, velocity, and impedance control. Different control modes can lead to  unpredictable latencies, which complicates the prediction of robot trajectories in videos.

Promising approaches have emerged to address these challenges,  although they are still in the early stages of development. Researchers  are exploring innovative methods to create more adaptable and unified  control frameworks for both robotics and gaming environments.  The [Genie](https://arxiv.org/abs/2402.15391) project designs a latent action space for video world models in  certain games, necessitating an adaptation process to translate latent  action embeddings into precise actions for each game. Similarly, the [Video Language Planning](https://video-language-planning.github.io/) project for robotics utilizes abstract actions  expressed in natural language as direct control signals for video  prediction models. This approach infers sequences of low-level actions  from predicted video sequences, supported by an inverse dynamics model.



### Lack of 3D information

Alternatively, Feifei Li's [World Labs](https://www.worldlabs.ai/) is building 3D models instead of video models to allow for interative 3D agent exploration. However, the 3D data collection is more expensive than videos. As a projection from 3D world, the 2D video model will suffer from theoretical hardness in fully modeling the dynamics in 3D. 

<img src="/home/quantumiracle/research/webpage/blogs/files/sora_robot.gif" alt="sora_robot" style="zoom:150%;" />

​       		<figcaption>A robot stacking blocks. Video source: OpenAI Sora model.</figcaption>

### Lack of multimodal information

haptics



## References

1. [Video Language Planning](https://video-language-planning.github.io/)
2. [Diffusion Forcing](https://boyuan.space/diffusion-forcing/)
3. [GameNGen](https://gamengen.github.io/)
4. [GameGen-x](https://gamegen-x.github.io/)
5. [Oasis](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model)
6. [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
7. [OpenAI's blog on Sora](https://openai.com/index/video-generation-models-as-world-simulators/)
8. [Open-Oasis](https://github.com/etched-ai/open-oasis)
9. [Diffusion Transformer](https://arxiv.org/abs/2212.09748)
10. [Loopy](https://arxiv.org/abs/2409.02634)
11. [HunyuanVideo](https://arxiv.org/abs/2306.04710)
12. [Classifier-free Guidance](https://arxiv.org/abs/2207.12598)
13. [ControlNet](https://arxiv.org/abs/2302.05543)
14. [Genie](https://arxiv.org/abs/2402.15391)
15. [World Labs](https://www.worldlabs.ai/)

