# Foundational Video World Model (2024.12 WIP)

A foundational video world model is crucial for advancements in robotics (e.g., [video language planning](https://video-language-planning.github.io/), [diffusion forcing](https://boyuan.space/diffusion-forcing/)) and the next generation of game engines (e.g., [GameNGen](https://gamengen.github.io/), [GameGen-x](https://gamegen-x.github.io/), [Oasis](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model)), autonomous driving, etc. In robotics, such a model could enable access to an infinite amount of interaction data within realistic environments, moving beyond the limitations of non-realistic simulators. This breakthrough has the potential to bypass the long-standing simulation-to-reality gap that has impeded the research community for over a decade.

<img src="/home/quantumiracle/research/webpage/blogs/files/vlp.gif" alt="vlp" style="zoom:150%;" />

![oasis](/home/quantumiracle/research/webpage/blogs/files/oasis.gif)

![gamegenx](/home/quantumiracle/research/webpage/blogs/files/gamegenx.gif)

[OpenAI's blog on Sora](https://openai.com/index/video-generation-models-as-world-simulators/) project claims video model as the world simulators. While this concept is undoubtedly inspiring and promising, it is important to approach such claims with skepticism. This blog critically examines the desiderata for constructing foundational video models that are both practical and feasible for robotics and the game industry. Drawing from the current state of video generation techniques, the limitations of existing models are highlighted. To guide the research community towards meaningful progress, I have identified examples for each issue as evidence, accompanied by further analysis to illuminate potential research directions.

## Desiderata
- Efficient video generation via few-step sampling, distillation, hardware acceleration.
- Long video generation with action condition: autoregressive
- unified latent action representation, to support diverse environments and agents. For game, it is the key-board mapping for player's control; for robotics, there are high-level abstract linguistic actions and low-level robot control actions. 
- Do we need depth information? if 3D information can be reconstructed with 2D videos or binocular vision, we may not necessarily require the depth information for foundational video world models.

## Problems

### Partial Observability

Partial observability has long been a challenge in reinforcement learning, as most practical environments provide agents with only incomplete information. This inherent limitation creates theoretical difficulties for agents in optimizing their policies, even with their best estimates of belief states.

Similarly, when a video generation model functions as a world simulator, it also faces the partial observability problem. To accurately reconstruct the dynamics and motions of agents and objects within an environment, the model must intake sufficient environmental information. Without such input, it cannot reliably predict future trajectories.

Drawing an analogy to large language models (LLMs), this issue resembles the challenge of addressing vague or underspecified user queries. Current solutions involve multi-turn question-and-answer interactions between users and LLMs to refine problem specifications. A similar approach could potentially mitigate ambiguity for video generation models. However, facilitating effective communication between the video generation model and the agent using it is considerably more complex.

Another method to address partial observability is by concatenating more historical frames to provide additional conditioning information for the video generation model. However, this approach imposes a heavier computational burden on the model, as the increased conditioning information will very likely add to the complexity of attention computations during subsequent frame generation.

### Recency bias

![mc_example](/home/quantumiracle/research/webpage/blogs/files/mc_example.gif)

The example is based on [open-oasis](https://github.com/etched-ai/open-oasis) project. The model is [diffusion transformer](https://arxiv.org/abs/2212.09748) with [diffusion forcing](https://arxiv.org/abs/2407.01392) technique to enable **autoregressive** rollout. The video diffusion model is conditioned on sequential action inputs. As shown in the example, the long video generation will collapse once the agent starts to stare at the ground, which causes the loss of information about surrounding objects and is no longer recovered. This is the recency bias of video world model due to lack of memory mechanism in video generation process. The partially observed recent historical frames do not encompass complete information about the environment, thus causing information loss in future frame prediction. [Loopy](https://arxiv.org/abs/2409.02634) project proposes to proportionally sampling (with recency weights) from all historical frames as additional condition for video generation, which alleviates the problem. But too many historical frames will increase the burden for diffusion modeling. Effective representation of historical information can be essential.

### Compounding error

![diffusionforcing](/home/quantumiracle/research/webpage/blogs/files/diffusionforcing.gif)

Different from text generation in large language models, video generation models suffer from the compounding error if applied in an autoregressive manner. By conditioning on previous frames, the frame-to-frame prediction error accumulates when iteratively sampling from a video model.

### Physical reality

![non_physical](/home/quantumiracle/research/webpage/blogs/files/non_physical.gif)

recent progress on this problem

### Lack of 3D information

Alternatively, Feifei Li's [World Labs](https://www.worldlabs.ai/) is building 3D models instead of video models to allow for interative 3D agent exploration. However, the 3D data collection is more expensive than videos.

### Lack of multimodal information

haptics

### Conclusion

Happy blogging!

