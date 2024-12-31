# Foundational Video World Model

Date: 2024.12.31 | Author: Zihan Ding



A foundational video world model is crucial for advancements in robotics (e.g., [video language planning](https://video-language-planning.github.io/), [diffusion forcing](https://boyuan.space/diffusion-forcing/)) and the next generation of game engines (e.g., [GameNGen](https://gamengen.github.io/), [GameGen-x](https://gamegen-x.github.io/), [Oasis](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model), [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)), autonomous driving, etc. In robotics, such a model could enable access to an infinite amount of interaction data within realistic environments, moving beyond the limitations of non-realistic simulators. This breakthrough has the potential to bypass the long-standing simulation-to-reality gap that has impeded the research community for over a decade.

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/vlp.gif" alt="vlp" style="zoom:200%;" /><figcaption>Video source: video-language-planning project page.</figcaption></figure>





<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/oasis.gif" alt="vlp" style="zoom:100%;" /><figcaption>Video source: open-oasis model.</figcaption></figure>





<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/gamegenx.gif" alt="vlp" style="zoom:100%;" /><figcaption>Video source: GameGen-x project page.</figcaption></figure>

[OpenAI's blog on Sora](https://openai.com/index/video-generation-models-as-world-simulators/) project claims video model as the world simulators. While this concept is undoubtedly inspiring and promising, it is important to approach such claims with skepticism. This blog critically examines the desiderata for constructing foundational video models that are both practical and feasible for robotics and the game industry. Drawing from the current state of video generation techniques, the limitations of existing models are highlighted. To guide the research community towards meaningful progress, I have identified examples for each issue as evidence, accompanied by further analysis to illuminate potential research directions.

## Desiderata
To establish a foundational video world model, the following desiderata must be addressed:

### * Fidelity and Realism

The model must accurately reflect physical laws for robotics or simulated rules in game engines. This fidelity ensures real-world applicability and effective simulation.

### * Efficient Video Generation

The model should support efficient video generation with minimal computational cost and latency. Techniques such as few-step sampling, distillation, and hardware acceleration are essential to achieve faster and more resource-efficient results.

### * Long Video Generation with Frame Conditioning

Long video sequences should be generated conditionally, leveraging previous or current frames as context. Autoregressive techniques are crucial to maintaining coherence over extended timelines.

### * Historical Information Retrieval

The ability to utilize historical frames or retrieval-augmented generation (RAG) is critical for context-aware generation. Alternative memory mechanisms can help maintain computational efficiency while effectively integrating past information.

### * Action Conditioning

The model should support agent interaction by processing action inputs and predicting future outcomes in video format. This capability is fundamental for applications in robotics and gaming.

### * Unified Latent Action Representation

The model must accommodate diverse environments and agents. For games, this involves mapping hardware inputs for standard user controls; for robotics, both high-level abstract actions and precise low-level robot control commands should be seamlessly integrated.

### * Controllability

A highly controllable model is essential for practical applications. Controllability allows for:

1. Reducing prediction uncertainty with external guidance.
2. Generating diverse samples under varying conditions.

Techniques such as [classifier-free guidance](https://arxiv.org/abs/2207.12598) and [ControlNet](https://arxiv.org/abs/2302.05543) enhancements offer promising ways to improve model controllability.

### * 3D Information

While incorporating 3D information can enhance realism, it may not always be necessary. Robust 3D reconstruction from 2D videos or binocular vision can serve as an efficient alternative. The depth information is provided in some cameras. If 3D information can be reconstructed with 2D videos or binocular vision, we may not necessarily require the depth information for foundational video world models.

## Problems

### Partial Observability

Partial observability has long been a challenge in reinforcement learning, as most practical environments provide agents with only incomplete information. This inherent limitation creates theoretical difficulties for agents in optimizing their policies, even with their best estimates of belief states.

Similarly, when a video generation model functions as a world simulator, it also faces the partial observability problem. To accurately reconstruct the dynamics and motions of agents and objects within an environment, the model must intake sufficient environmental information. Without such input, it cannot reliably predict future trajectories.

Drawing an analogy to large language models (LLMs), this issue resembles the challenge of addressing vague or underspecified user queries. Current solutions involve multi-turn question-and-answer interactions between users and LLMs to refine problem specifications. A similar approach could potentially mitigate ambiguity for video generation models. However, facilitating effective communication between the video generation model and the agent using it is considerably more complex.

Another method to address partial observability is by concatenating more historical frames to provide additional conditioning information for the video generation model. However, this approach imposes a heavier computational burden on the model, as the increased conditioning information will very likely add to the complexity of attention computations during subsequent frame generation.

### Recency Bias

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/mc_example.gif" alt="vlp" style="zoom:200%;" /><figcaption>Video generated with open-oasis model.</figcaption></figure>

Recency bias in video generation models is closely linked to the lack of long-term memory mechanisms in their design. A relevant example as above can be observed in the [Open-Oasis](https://github.com/etched-ai/open-oasis) project, where the model employs a [Diffusion Transformer](https://arxiv.org/abs/2212.09748) with the [Diffusion Forcing](https://arxiv.org/abs/2407.01392) technique to enable autoregressive rollouts. In this approach, the video diffusion model is conditioned on sequential action inputs.

As demonstrated in the example, long video generation often collapses when the agent begins to focus solely on the ground, losing critical information about surrounding objects. This information loss is not recoverable without sufficient historical information, exemplifying the recency bias in video world models caused by the absence of memory mechanisms in the video generation process. Because partially observed recent historical frames do not capture the complete environmental context, the model suffers from incomplete information, leading to inaccuracies in future frame predictions.

The [Loopy](https://arxiv.org/abs/2409.02634) project proposes addressing this issue by proportionally sampling, with recency weights, from all historical frames as additional conditions for video generation. While this approach alleviates recency bias, incorporating too many historical frames imposes a significant computational burden on diffusion modeling. Thus, effectively representing historical information remains critical for overcoming these challenges in video generation.

### Compounding Error

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/diffusionforcing.gif" alt="vlp" style="zoom:100%;" /><figcaption>Video source:         <a href="https://arxiv.org/abs/2407.01392" target="_blank" rel="noopener noreferrer">Diffusion Forcing</a> project page.</figcaption></figure>

To generate sequences of arbitrary or extra-long length, video models need to be sampled in an autoregressive manner. However, autoregressive sampling in continuous space often accumulates small errors at each step, leading to samples that deviate significantly from the original distribution—a phenomenon known as *compounding error*. In robotics planning, compounding error is a well-known issue when dynamics models with approximation errors are applied autoregressively. Since the essence of video models is to predict future states in visual format based on current states, they inherently fall under the category of dynamics models and are thus susceptible to compounding error.

Unlike text generation with discrete tokens in large language models, video generation models are particularly prone to significant compounding errors when applied autoregressively. By conditioning predictions on previous frames, frame-to-frame prediction errors accumulate during iterative sampling. Within this framework, the Teacher Forcing method, as discussed earlier, clearly suffers from severe compounding error, as shown in above video examples. In contrast, [Diffusion Forcing](https://arxiv.org/abs/2407.01392), which incorporates denoising across various noise levels within a predicted sequence, has been shown to mitigate compounding error when used in an autoregressive manner. The issue of compounding error in long video generation remains largely underexplored.

### Physical Reality

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/non_physical.gif" alt="vlp" style="zoom:100%;" /><figcaption>Video source: OpenAI Sora model.</figcaption></figure>

Since the initial public announcement of the Sora project, it has been widely acknowledged that current video generation methods using diffusion models cannot fully guarantee physical reality, as shown in above video example. To date, no principled solution has been established to address this limitation. In game simulation, this issue is less critical compared to robotics video modeling, where adherence to the laws governing our physical world is highly influential for robot planning performances. While game simulations primarily demand visual realism (not even physical reality), existing techniques still fall short of ensuring that no artificial or unnatural processes occur during the simulation.

On one hand, practical observations confirm that larger models trained on more realistic datasets tend to generate video samples with greater fidelity and fewer observable artifacts. It is reasonable to anticipate that this issue could become less pronounced when models are trained on datasets several orders of magnitude larger. A successful analogy can be drawn from the ability of large language models (LLMs) to master linguistic grammar.

On the other hand, compared to linguistic grammar, physical laws are significantly more complex and structured, with an almost zero tolerance for errors. This complexity makes the emergence of a model capable of fully grasping physical laws purely from video data seem somewhat improbable. For example, LLMs, despite their advances, still struggle to fully understand mathematics or even basic arithmetic. This highlights a dual challenge: the difficulty LLMs face with mathematics parallels the challenge of capturing physics in foundational video models.

[A case study](https://phyworld.github.io/) on diffusion-based video generation models has demonstrated that scaling alone may not suffice to address the lack of physical laws in video models. Real-world dynamic processes are inherently complex, influenced by variations in object physical properties, geometries, the number of interacting objects, external forces, and backgrounds. This leads to that:

**Real-world dynamic processes usually exhibit combinatorial complexity.**

As a result, video models tend to perform well on in-distribution cases but struggle with out-of-distribution scenarios governed by the same physical laws. This discrepancy highlights their inability to fully internalize and generalize physical principles. An example is provided below, showing a misalignment in motion between ground-truth videos (top) and generated videos (bottom) in out-of-distribution cases.

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/simple_failure_cases.gif" alt="vlp" style="zoom:150%;" /><figcaption>Videos from         <a href="https://phyworld.github.io/" target="_blank" rel="noopener noreferrer">project page</a>.</figcaption></figure>

Research efforts such as [PhysGen](https://stevenlsw.github.io/physgen/) and [PhysDreamer](https://physdreamer.github.io/) explicitly incorporate physical parameters or models into the video generation process, aiming to enhance the physical reality of synthesized object motions.

### Unnatural Motion

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/hunyuan_cat.gif" alt="vlp" style="zoom:100%;" /><figcaption>A cat walks on the grass, realistic style, by     <a href="https://arxiv.org/abs/2306.04710" target="_blank" rel="noopener noreferrer">HunyuanVideo</a> model.</figcaption></figure>

Existing video generation models struggle significantly with  multi-object scenarios and fast motion, particularly when similar visual patterns occur in close proximity within a small spatial region across  frames. Examples include the rapid movements of human fingers, animal  legs, and entire animal bodies (as illustrated in the example above).  The primary challenge arises from the diffusion model's reliance on a  denoising process. In this context, nearby pixels that contain random noises tend to exhibit similarities in the latent noised space, leading to ambiguity for the diffusion model. This ambiguity hinders its ability to effectively differentiate between closely related yet distinct denoising trajectories.

### Controllability

Controllability is a crucial aspect of practical world simulators, extending beyond the generation of outcomes based solely on training distributions or  deterministic realistic trajectories. Due to inherent limitations in  training data coverage or insufficient information, model predictions  should maintain a certain level of uncertainty. In this context, the  controllability of the model serves at least two essential purposes:

1. Uncertainty Reduction: With additional guidance  provided, a highly controllable model should be able to minimize  uncertainty in future predictions.
2. Sample Diversity: Given various forms of guidance, the controllable model should generate diverse samples.

From an application perspective, controllable models offer  significant advantages. In game simulation, they can generate futures  with different properties, such as object layouts or environmental  states, based on various provided conditions. For robotic simulation,  enhanced controllability leads to more precise trajectory predictions.  Furthermore, the ability to introduce various conditions in controllable models can serve a purpose similar to domain randomization,  facilitating the development of robust robotic policies.

The key to achieving controllability lies in enhancing the model's dependency on externally provided additional conditions for conditional modeling. Current techniques for controllable generation include  [classifier-free guidance](https://arxiv.org/abs/2207.12598) in pre-training diffusion models and  architectural designs like [ControlNet](https://arxiv.org/abs/2302.05543), which improve controllability  during the fine-tuning process. From the aspect of architectural design of the [diffusion transformer (DiT)](https://arxiv.org/abs/2212.09748), the conditioning mechanism can be integrated through: (1) self-attention applied to a fully concatenated sequence of both the conditioning data and the video sequence, (2) cross-attention mechanisms that directly attend from the video sequence to the conditioning sequence, or (3) adaptive LayerNorm, which injects the conditional signals into the network.

These approaches represent significant strides towards creating  more versatile and adaptable world simulators. However, as the field  progresses, further research is needed to refine these methods and  explore new avenues for enhancing model controllability across diverse  applications.

### Diverse Control Space

To establish a foundational video world model, it is essential that the  model accommodates diverse control mechanisms. This challenge is evident in both robotics and gaming video models. In gaming engines, the action space varies significantly from one game to another due to differences  in control design and hardware interfaces. For instance, in the game *Street Fighter*, apart from standard motions, players can perform at least three distinct actions for kicks and three for punches, along with additional special moves and action  combinations. Conversely, other games may lack these specific actions or exhibit substantial variations in their implementation. Furthermore,  the unification of hardware interfaces presents another challenge, as  users may input commands through keyboards, mice, gamepads, or other  devices. 

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/sf_control_pad.png" alt="vlp" style="zoom:50%;" /><figcaption>Control pad for StreetFigher games.</figcaption></figure>

In robotics, the challenge of developing a unified control space  for foundational world models is exacerbated by the unique embodiments  of different robots, which result in varying degrees of freedom and  distinct action spaces. Each robot's design influences its control  mechanisms, leading to differences in latency and control modes, such as position, velocity, and impedance control. Different control modes can lead to  unpredictable latencies, which complicates the prediction of robot trajectories in videos.

Promising approaches have emerged to address these challenges,  although they are still in the early stages of development. Researchers  are exploring innovative methods to create more adaptable and unified  control frameworks for both robotics and gaming environments.  The [Genie](https://arxiv.org/abs/2402.15391) project designs a latent action space for video world models in  certain games, necessitating an adaptation process to translate latent  action embeddings into precise actions for each game. Similarly, the [Video Language Planning](https://video-language-planning.github.io/) project for robotics utilizes abstract actions  expressed in natural language as direct control signals for video  prediction models. This approach infers sequences of low-level actions  from predicted video sequences, supported by an inverse dynamics model.

### 3D Consistency

As a projection from the 3D world, 2D video models inherently face theoretical challenges in fully capturing the spatial relationships and dynamics of three-dimensional spaces. Humans, by contrast, possess the innate ability to subconsciously construct mental representations of spatial environments—a cognitive skill referred to as *spatial intelligence*. This concept, rooted in psychology, has recently been highlighted in the work of  Feifei Li's [World Labs](https://www.worldlabs.ai/). The lack of 3D consistency relates to the broader issue of previously discussed physical reality but differs in that it may not necessarily require an understanding of physical laws. Instead, it primarily involves grasping the spatial structure of the world.

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/space_reason.gif" alt="vlp" style="zoom:150%;" /><figcaption>Real video from       <a href="https://vision-x-nyu.github.io/thinking-in-space.github.io/" target="_blank" rel="noopener noreferrer">project</a>.</figcaption></figure>

Visual understanding forms the foundation of spatial intelligence. Two main categories of methods exist for achieving visual understanding: (1). **Multimodal language models**. These follow strategies such as [Locked-image text tuning (LiT)](https://arxiv.org/abs/2111.07991), which fine-tune the text encoder to align with a frozen vision encoder pre-trained via self-supervised learning. Examples of this approach include [LLaVA](https://arxiv.org/abs/2304.08485) and [DINO.txt](https://www.arxiv.org/abs/2412.16334). (2). **Video sequence prediction models**. These focus on predicting future outcomes within the same visual format, as extensively discussed in previous paragraphs. These two approaches differ in their treatment of visual understanding. The first provides a linguistic representation of visual inputs, while the second predicts future visual sequences. Despite their differences, neither approach has yet achieved the level of spatial intelligence required to serve as a foundational world model.

A critical question arises: **do current multi-modal models or video models develop an internal world representation through learning from text-visual embedding alignment or video sequence predictions**? The evidence suggests otherwise. [Recent research](https://vision-x-nyu.github.io/thinking-in-space.github.io/) on multimodal large language models (LLMs) indicates that these models underperform significantly compared to humans in reasoning about spatial information derived from videos.

Similarly, video sequence prediction models struggle to maintain consistent 3D information across successive 2D video frames. This limitation becomes evident when assessing whether such models can accurately predict videos that faithfully reconstruct 3D spatial information under arbitrary camera motions—particularly when provided with a global view of the environment, considering the above real video example. A failure in this regard underscores the absence of a robust internal 3D world model in the current approach.

In robotics foundation models, the absence of 3D information significantly impairs performance across various tasks, including 3D navigation, object localization, and dexterous manipulation. For example, consider the generated video samples (shown below) of cube manipulation using robotic arms. Accurate 3D information, encompassing object-object and object-agent interactions, is crucial for these tasks. It facilitates precise gesture estimation, prevents object penetration in predicted trajectories, and ensures consistency in spatial reasoning. These factors are critical for achieving success in dexterous manipulation, where subtle inaccuracies can compromise the entire operation.

An alternative and promising direction is the development of 3D world models, enabling interactive 3D agent exploration. However, this approach comes with a significant drawback: The collection of 3D data is significantly more resource-intensive than that of 2D video data, which can be captured using a single camera and offers access to virtually unlimited data resources in the physical world.

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files/sora_robot.gif" alt="vlp" style="zoom:150%;" /><figcaption>A robot stacking blocks. Video source: OpenAI Sora model.</figcaption></figure>

### Multimodality

In addition to text and visual information, other modalities such as haptics and audio are crucial for specific tasks in the application of foundation models. For instance, in dexterous manipulation, humans heavily rely on haptic feedback from their hands to achieve stable grasping and perform fine-grained manipulations of complex objects and tasks. However, research on incorporating these additional modalities into foundation models remains underexplored and holds significant potential for advancing their capabilities.

## Citation

```
@article{ding2024foundational,
  title   = "Foundational Video World Model",
  author  = "Ding, Zihan",
  journal = "quantumiracle.github.io",
  year    = "2024",
  month   = "Dec",
  url     = "https://quantumiracle.github.io/webpage/blogs/blog20241231.html"
}
```



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
11. [How Far is Video Generation from World Model](https://phyworld.github.io/)
12. [PhysGen](https://stevenlsw.github.io/physgen/)
13. [PhysDreamer](https://physdreamer.github.io/)
14. [HunyuanVideo](https://arxiv.org/abs/2306.04710)
15. [Classifier-free Guidance](https://arxiv.org/abs/2207.12598)
16. [ControlNet](https://arxiv.org/abs/2302.05543)
17. [Genie](https://arxiv.org/abs/2402.15391)
18. [World Labs](https://www.worldlabs.ai/)
19. [Locked-image text tuning (LiT)](https://arxiv.org/abs/2111.07991)
20. [LLaVA](https://arxiv.org/abs/2304.08485)
21. [DINO.txt](https://www.arxiv.org/abs/2412.16334)
22. [Think in Space](https://vision-x-nyu.github.io/thinking-in-space.github.io/)

