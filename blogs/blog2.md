# Nuts and Bolts in Training Transformers and Diffusion Models

Date: 2025.04.15 | Author: Zihan Ding

Recent advances in machine learning have introduced powerful innovations – from **infinite-context Transformers** and new **normalization-free networks**, to novel **diffusion model architectures**, alternative **flow-matching training objectives**, and optimized **tokenization schemes**. To understand how these ideas perform in practice, I conducted a series of experiments. This post presents key findings and analysis from those experiments, focusing on *why* certain designs behaved as they did. We explore four areas: **(1)** Transformers with Infini-Attention and Dynamic Tanh, **(2)** Diffusion Models (using both Transformer and U-Net backbones), **(3)** Flow Matching and Shortcut generative models, and **(4)** Tokenizer design for latent diffusion. The goal is to provide insights and lessons for researchers and practitioners looking to build upon these techniques. 

## Transformers

### Infini-Attention vs. Vanilla

One experiment tackled extending Transformer sequence length with **Infini-Attention**, a mechanism that enables *unbounded context* via a compressive memory buffer ([Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)). Infini-Attention combines local attention on recent tokens with a *long-term memory* of older tokens that is continually compressed, theoretically allowing **infinite context length** without unbounded memory growth, however the practical experiments show compromised results ([A failed experiment: Infini-Attention, and why we should keep trying?](https://huggingface.co/blog/infini-attention#:~:text=Motivated%20by%20this%2C%20we%20explore,In%20theory%2C%20we)). In our trials on sequence modeling, we found that while Infini-Attention worked in principle, its **training convergence was significantly slower** than a vanilla Transformer with the same configuration. This experiment is conducted on *enwik8* dataset for next-token prediction. 

<figure><img src="https://quantumiracle.github.io/webpage/blogs/files2/doc_img_000.png" alt="vlp"/>

**Figure 1** illustrates the training loss curves: the Infini-Attention model’s loss decreases much more slowly, taking many more updates to approach the performance of the baseline Transformer. 

This suggests that the added complexity of compressing and retrieving long-term contexts makes optimization harder, at least on the data/tasks we used. Notably, when we enabled the gradient with backpropagation-through-time (BPTT), it had minimal effect on Infini-Attention’s performance – truncated or full backpropagation yielded nearly the same results. This implies that the compressive memory effectively preserves long-term information such that reducing BPTT length doesn’t hurt learning (and also that simply extending BPTT doesn’t solve the slow convergence). Our experience echoes a report from Hugging Face’s team: replicating Infini-Attention required extensive debugging due to convergence difficulties ([A failed experiment: Infini-Attention, and why we should keep trying?](https://huggingface.co/blog/infini-attention#:~:text=While%20understanding%20a%20new%20method,and)). This indicates **Infini-Attention may need careful tuning (learning rates, initializations, etc.) to train as fast as standard attention**, and future work should aim to improve its optimization. Nonetheless, once trained, it offers the enticing benefit of processing arbitrarily long sequences within a fixed memory budget ([A failed experiment: Infini-Attention, and why we should keep trying?](https://huggingface.co/blog/infini-attention#:~:text=2024%20,in%20reality%20compression%20limits%20the)) – a capability increasingly important for long-document and streaming applications.

### Normalization-Free Networks

![](assets_doc/doc_img_001.png)

**Figure 2** illustrates the training loss curves for *Dynamic Tanh* (DyT) and RMSNorm (vanilla)

Another Transformer experiment evaluated **normalization-free networks** using *Dynamic Tanh* (DyT) in place of standard normalization layers. Recent work by Zhu et al. (2025) showed that a simple scaled Tanh activation can substitute for LayerNorm/RMSNorm, enabling Transformers *without* normalization layers to match or exceed the performance of normalized ones ([Transformers without Normalization](https://arxiv.org/abs/2503.10622)). We replaced RMSNorm with Dynamic Tanh in the vanilla Transformer and observed *no significant difference* in the training dynamics or final loss – the loss curves for DyT vs. RMSNorm were nearly identical. This result supports the claim that **Dynamic Tanh can stabilize training comparably to RMSNorm**, without the overhead of computing normalization statistics. In practice, removing normalization can slightly improve speed and simplify architecture, so it’s promising that DyT worked as a drop-in replacement in our test. Our findings align with the conclusions of Zhu et al., who demonstrated Transformers with DyT achieving on-par accuracy with normalized counterparts across various tasks ([Transformers without Normalization](https://arxiv.org/html/2503.10622v1#:~:text=Normalization%20layers%20are%20ubiquitous%20in,We%20validate%20the)). Overall, this suggests that future Transformer designs might omit normalization layers entirely, using techniques like Dynamic Tanh to maintain stability while gaining a small efficiency boost. However, the efficiency boost can depend on the architecture implementation and computational hardware.

## Diffusion Models

Diffusion models have become a cornerstone of generative modeling. We experimented with diffusion on the MNIST image dataset under two settings: using a **Transformer-based backbone (Diffusion Transformer, or DiT)** and using the conventional **U-Net backbone**. Our experiments aimed to see how various design choices affect training convergence and sample quality.

### Diffusion Transformers (DiT)

We built a **Diffusion Transformer (DiT)** model for image generation, broadly following the idea of replacing the usual U-Net with a Transformer encoder-decoder as the denoising network. Several hyperparameters and architectural choices were tested:

- **Number of Diffusion Timesteps:** We compared using the standard ~$1000$ timesteps (as in DDPM ([experiment.md](file://file-Ei3Q8tCqFntiXVJhWvjZrN#:~:text=DDPM%20timesteps))) versus greatly reduced counts (e.g. 400 or 40 steps). We found that using **1000 diffusion steps led to faster convergence and a lower final loss** than fewer steps. In fact, on MNIST the model with 1000 steps not only reached a slightly better loss than the 400-step model, but *significantly* outperformed a model trained with only 40 steps. Intuitively, a diffusion process with more timesteps means each denoising step is smaller and easier to learn, which likely makes optimization simpler – the model can focus on removing a tiny bit of noise at a time. With too few steps, each step must denoise a large amount of noise, which proved much harder for the Transformer to learn (the 40-step model struggled to reconstruct digits clearly). This observation is consistent with the original DDPM work that employed 1000 steps, and it mirrors the trade-off in diffusion models: more steps improve generation fidelity but at the cost of slower sampling. 

![](assets_doc/doc_img_002.png)**Figure 3** shows the training loss curves for different step counts – the 1000-step curve dives down fastest and ends lowest, while the 40-step curve converges to a higher loss. In practice, one might use advanced samplers to reduce sampling time, but when training from scratch, having a sufficiently fine discretization of the diffusion process is clearly beneficial.



- **Patch Size:** In our DiT, images are split into patches that become token inputs to the Transformer, following Vision Transformer practice. We experimented with different patch sizes for 28×28 grayscale MNIST (patch sizes 1×1 vs 2×2). The results were striking – using **1×1 patches (i.e. each pixel as a token) was critical to get good performance**, whereas 2×2 patches caused a significant drop in generation quality. Essentially, a 2×2 patch means each token represents a 4-pixel block of the image; for MNIST digits (which have fine details like thin strokes), this coarse patchification loses important information. The model with 2×2 patches often produced blurry or incomplete digits, indicating underfitting. In contrast, the model with 1×1 patches (effectively treating the image like a 784-length sequence of pixels) learned to generate clearly recognizable digits, albeit with a slightly lower fidelity than an equivalent U-Net. This suggests that **for high-resolution or detail-critical data, the patch size must be small enough to capture essential local structure**. Our finding aligns with the general understanding that Vision Transformers require appropriately sized patches – too large patches reduce the granularity of input features. (For instance, recent latent diffusion models use patch-like tokenizers but operate on high-level feature maps where some loss of spatial detail is acceptable ([Reconstruction vs. Generation](https://arxiv.org/pdf/2501.01423)). In the case of raw MNIST pixels, any patch larger than 1×1 was detrimental. In future work on DiT for higher-resolution images, one might incorporate convolutional preprocessing or hierarchical tokens to avoid this issue, or use learned tokenizers that compress without losing critical detail.

  ![](assets_doc/doc_img_003.png)

  **Figure 4** shows the training loss curves for different patch size for MNIST image generation.

  ![](assets_doc/doc_img_004.png)

  **Figure 5** shows patch size 2x2 sampling results.

  

  ![](assets_doc/doc_img_005.png)

  **Figure 6** shows patch size 1x1 sampling results.

  

- **Class Conditioning Format:** We also explored how to feed class labels into the conditional diffusion model. MNIST has 10 classes (digits 0–9). One approach is **one-hot encoding** the class (a 10-dimensional vector with a 1 at the class index) and injecting it into the model (e.g. concatenating to token embeddings or through a conditioning layer). Another approach is to use a single integer label and let the model internally embed it (essentially a learned embedding lookup). In our experiments, the **one-hot conditioning dramatically outperformed the single embedded label**. With one-hot conditioning, the generated samples correctly and sharply reflected the conditioning class. When we used an integer label with an embedding (same embedding dimension), the model’s outputs were much less accurate to the class and often of lower visual quality. We suspect that one-hot encoding provided a richer, high-signal input for each class – effectively giving the network 10 “channels” to distinctly activate for each class – whereas a learned embedding might be harder to propagate through the Transformer or might not disentangle class information as well in a small model. This is a noteworthy insight: **for conditional diffusion on a small-scale problem, a simple one-hot conditioning vector can be a better choice than a learned class embedding, although barely reflected from the training curves**. It ensures the conditioning information is explicit and easily accessible to every part of the network. In larger models (e.g. big image-generation transformers or language models), learned embeddings are standard, but our result suggests that at least for small architectures or when data is limited, one-hot features may ease the learning of conditioning. (It’s akin to how some small CNNs benefit from one-hot inputs in early layers instead of a single label embedding.)

  ![](assets_doc/doc_img_006.png)

  **Figure 7** shows the training loss curves for one-hot vs. integer class condition for DiT on MNIST image generation.

  ![](assets_doc/doc_img_005.png)

  **Figure 8** shows one-hot condition sampling results.

  ![](assets_doc/doc_img_007.png)

  **Figure 9** shows integer (no one-hot) condition sampling results:

  

- **Model Depth:** We trained DiT models with different numbers of Transformer blocks (layers) to see the effect of network depth. Not surprisingly, **a deeper Transformer yielded better performance** – e.g. a 6-block DiT achieved lower loss and generated digits with fewer artifacts compared to a 4-block DiT (which in turn was better than a 2-block version, and so on). This trend is consistent with the general understanding that increasing model capacity (to a point) improves generative performance. A deeper model can capture more complex patterns and has higher expressive power. In our case, going from 4 to 6 layers made a noticeable difference in output clarity. Of course, deeper models train slower and are more memory intensive; one must balance resources. But given the improvements we saw, scaling up depth is a straightforward way to get better diffusion model results. This mirrors observations in the literature that **larger diffusion models produce higher fidelity samples** (for example, Guided Diffusion improved class-conditional ImageNet generation by scaling up model size along with other tweaks). In practice, one might combine depth with other efficiency tricks (like gradient checkpointing or sparse attention) to manage the compute cost.

  ![](assets_doc/doc_img_008.png)

  **Figure 10** shows the training loss curves for different numbers of diffusion blocks for DiT on MNIST image generation.

  ![](assets_doc/doc_img_009.png)

  **Figure 11** shows depth-4 DiT sampling results.

  ![](assets_doc/doc_img_005.png)

  **Figure 12** shows depth-6 DiT sampling results.

  

- **Rotary Embedding Frequency:** Our DiT used Rotary Positional Embeddings (RoPE) for spatial position encoding in self-attention. RoPE has a hyperparameter defining the base frequency of rotation (which effectively controls at what spatial scale the rotations complete a full cycle). We tried two settings (base frequency corresponding to ~64 pixels vs ~256 pixels). **The choice of RoPE frequency had negligible impact on MNIST generation.** The training and samples were virtually the same for both settings. This isn’t too surprising – MNIST images are small, so as long as the positional encoding meaningfully distinguishes positions across 28×28, the exact periodicity beyond that scale doesn’t matter much. Both 64 and 256 are larger than 28, so they behave similarly in this regime. It’s reassuring that the model is not very sensitive to this hyperparameter here. For much larger images or different tasks like long video generation, tuning RoPE frequency might matter (as it could affect how well the model can attend across spatial distances), but in our case it was not a concern. This is also how RoPE scaling ([Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)) can be used to extend the context length for transformers.

- **2D vs. 3D Attention:** We conducted an interesting variant of the DiT: treating the input as a trivial “video” with an added time/frame dimension of size 1. In other words, we fed the images as shape (B, C, H, W) normally, and also as (B, **1**, C, H, W) to a **3D Transformer** that performs self-attention across height, width, *and* the extra temporal dimension. The idea was to simulate a video-diffusion Transformer (which attends over space and time) even though we only have one frame – essentially to test if the architecture could generalize to an extra dimension without issue. We applied *sinusoidal* positional embeddings for the frame dimension. As expected, the 3D DiT functioned correctly, but it was much heavier computationally: we had to halve the batch size due to increased memory use (since attention now considers an extra dimension). In terms of results, **the 3D-attention model did not show any clear advantage over the standard 2D-attention model** on single-frame data. The sample quality and loss were similar (once we adjusted for the smaller batch). This is reassuring in that the Transformer can seamlessly incorporate an extra spatial dimension, but it also confirms that if there is no actual variability in that dimension (only one frame), you don’t gain anything by attending over it. In a real video generation scenario with multiple frames, a 3D DiT could model temporal correlations directly. Our little experiment serves as a sanity check: introducing a dummy extra dimension doesn’t break the model (just slows it down), and by extension we anticipate that a true multi-frame (video) diffusion Transformer would behave reasonably – albeit with significant memory costs. (Notably, recent video diffusion models often combine 2D spatial attention with temporal attention in blocks, rather than full 3D attention everywhere ([Benchmarking and Improving Video Diffusion Transformers For ...](https://arxiv.org/html/2503.17350v1#:~:text=Benchmarking%20and%20Improving%20Video%20Diffusion,Furthermore%2C)), to save compute.)

  ![](assets_doc/doc_img_011.png)

  **Figure 13** shows the training loss curves for 2D DiT  and 3D DiT on MNIST image generation.

  ![](assets_doc/doc_img_005.png)

  **Figure 14** shows 2D DiT sampling results.

  ![](assets_doc/doc_img_010.png)

  **Figure 15** shows 3D DiT sampling results.

  

### Diffusion U-Net

To ground our findings, we also ran diffusion experiments using the **U-Net architecture** (as DDPM and Stable Diffusion backbones). These trials on MNIST let us compare against the Transformer approach and examine guidance techniques:

- **Diffusion Steps (U-Net):** Just as with the DiT, we tried training a U-Net diffusion model with 1000, 400, and 40 timesteps. The trend was the same: **1000 timesteps yielded the best and fastest-converging results**, 400 was slightly worse in loss but similar for sampling performances, and 40 was markedly worse. Even for the convolutional U-Net, reducing the number of diffusion steps made it harder to model the data distribution. After the same number of training epochs, the 1000-step U-Net had the lowest loss, the 400-step model a bit higher, and the 40-step model highest. The final sample quality reflected this: with 1000 steps, the generated digits were very crisp; with 400 they were mostly good with an occasional flaw; with only 40 steps, many samples were fuzzy or looked like interpolations between multiple digits. Clearly, **the U-Net needs a sufficiently fine diffusion process to capture the data well** – short processes (40 steps) did not give it enough gradual refinement stages, so generation quality suffered. This mirrors what we saw in the Diffusion Transformer case, suggesting that the requirement for many steps is a general phenomenon for diffusion loss, not specific to model type. It’s worth noting that there are methods to **speed up sampling** at inference (e.g. skipping steps or using learned samplers), but those usually still train with a large T (often 1000) to learn the detailed score function before distilling it down. Our results reaffirm that if one tries to train a diffusion model with too few steps from the start, it’s difficult to reach the same quality as a longer process.

  ![](assets_doc/doc_img_012.png)

  **Figure 16** shows the training loss curves for different step counts with Diffusion U-Net.

  ![](assets_doc/doc_img_013.png)

  **Figure 17** shows Diffusion U-Net 1000-step sampling results.

  ![](assets_doc/doc_img_014.png)

  **Figure 18** shows Diffusion U-Net 400-step sampling results.

  ![](assets_doc/doc_img_015.png)

  **Figure 19** shows Diffusion U-Net 40-step sampling results.

  

- **Classifier-Free Guidance (CFG):** We experimented with **guidance scale** during inference, a technique introduced by Ho & Salimans (2022) that trades off diversity for fidelity in conditional diffusion models ([Classifier-free Guidance with Adaptive Scaling - arXiv](https://arxiv.org/html/2502.10574v1#:~:text=Classifier,over%20the%20generative%20process)). In classifier-free guidance, one uses an unconditional model in combination with the conditional model to push samples toward the conditioning signal. The *guidance weight* (often denoted $w$ or $s$) determines how strongly the model leans into the conditional signal. We generated samples from the class-conditional U-Net with a high CFG weight (e.g. $w=2.0$) versus a low weight ($w=0.5$). As expected, **larger guidance weights produced much clearer and more class-consistent digits**, while low weights led to more muddled outputs. For instance, with $w=2.0$, when asking for a “7” the model produced a sharp, well-defined 7; at $w=0.5$, the result might appear ambiguous between a 7 and something else (or just blurrier). This aligns perfectly with known behavior of classifier-free guidance: *increasing the guidance scale dramatically improves sample fidelity*, making the outputs more closely match the condition, at the cost of some diversity (and potentially introducing minor artifacts if pushed too high). In our MNIST case, diversity isn’t a big concern (each class has a fairly narrow range of outputs), so a higher CFG weight was strictly beneficial for visual quality. The takeaway is that **using sufficient guidance is important for conditional diffusion** – it can significantly enhance the clarity of samples. One should be mindful to find a good range (in our tests, values around 1.5–2.5 were best; extremely high values can sometimes distort outputs). 

  ![](assets_doc/doc_img_014.png)

  **Figure 20** shows Diffusion U-Net 400-step with CFG $w=2.0$ sampling results.

  

  ![](assets_doc/doc_img_016.png)

  **Figure 21** shows Diffusion U-Net 400-step with CFG $w=0.5$ sampling results.

  

- In summary, our diffusion model experiments underscore a few points: (1) **the importance of a finely discretized diffusion process** (more steps yield better results, all else equal) for both DiT and Diffusion U-Net, (2) **the need for appropriate input representations** (patch size or tokens must retain important info; too aggressive compression hurts), (3) **the benefit of explicit conditioning signals** (one-hot class vectors, guidance weighting), and (4) **scaling model capacity** (depth) improves generative performance. Actually, for this MNIST experiment, I felt the Transformer-based diffusion slightly lagged a well-tuned U-Net on this task. In practice, U-Nets are highly optimized for images, while Transformers offer flexibility (e.g. easier multi-modality, longer-range interactions). It will be interesting to see if future *Diffusion Transformers* can close the gap with more refined training techniques or hybrid architectures – for example, one could combine a multi-layer convolutional encoder for patch extraction with a Transformer decoder, to get the best of both worlds.

## Flow Matching: An Alternative to Diffusion

### Flow Matching

**Flow Matching (FM)** is a recently proposed paradigm for generative modeling that forgoes simulating the diffusion process during training ([Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)). Instead of learning to denoise step-by-step, Flow Matching trains a continuous normalizing flow by directly **regressing the velocity field** (the ODE or “flow” that transforms noise into data) along a chosen probability path from noise to data. In simpler terms, the model is trained to *match* a prescribed continuous probability “flow” without needing to iteratively generate sample trajectories during training (hence “simulation-free”). This framework can be unified with diffusion as a special case, see [Generative Diffusion Modeling: A Practical Handbook](https://arxiv.org/abs/2412.17162).

We implemented a **Flow Matching Transformer** (essentially a Transformer-based continuous flow model) on MNIST and compared it with the Diffusion Transformer (DiT) from earlier. A striking result was that **the flow matching model achieved comparable performance to the DiT** in roughly half the training epochs. For example, after only 10 epochs of training, the flow model’s sample quality and loss were about on par with the DiT’s performance at 20 epochs. By 20 epochs, the flow model slightly surpassed the DiT. This suggests that **Flow Matching can learn the generative mapping faster** than the diffusion training procedure in our setup. One reason could be that FM optimizes a simpler objective (a direct regression of the continuous-time score field) without the stochastic noise of simulating many discrete diffusion steps. Chen et al. (2023) reported that using flow matching with appropriate probability paths yields more stable and data-efficient training than diffusion, and our experiment supports the notion that *fewer epochs* may be needed to reach a given likelihood or quality level. This is encouraging because faster convergence directly translates to less compute for model training. 

![](assets_doc/doc_img_017.png)

**Figure 22** shows the training loss curves for Flow Matching Transformer.

![](assets_doc/doc_img_018.png)

**Figure 23** shows sampling results after 10 epochs  for Flow Matching Transformer.

![](assets_doc/doc_img_019.png)

**Figure 24** shows sampling results after 20 epochs for Flow Matching Transformer.

![](assets_doc/doc_img_020.png)

**Figure 25** shows sampling results after 10 epochs for DiT.

![](assets_doc/doc_img_005.png)

**Figure 26** shows sampling results after 20 epochs for DiT.



### FasterDiT

We also tested some tweaks to the flow matching objective to potentially improve training efficiency. In [FasterDiT](https://arxiv.org/abs/2410.10356), they proposed to improve the efficiency for training using the **$v$-direction** loss which additionally minimizes the directional divergence of predicted velocity and ground truth, and we also experimented with re-weighting the time variable (using a log-normal distribution of time steps during training, referred to as *log-norm t* in our notes), as well as input standard deviation scaling trick. The paper suggests that these changes can further speed up convergence or improve quality. **The outcome, however, was that these modifications did not significantly change the performance**. In our simple experiments, the **flow model with $v$-direction loss, log-time weighting and standard deviation scaling** (with target 0.82 as suggested by the paper), trained to almost the same loss as the standard flow model. 

![](assets_doc/doc_img_022.png)

**Figure 27** shows the training loss curves for FasterDiT (based on flow matching), with $v$-direction loss, *log-norm t* and standard deviation scaling tricks.

![](assets_doc/doc_img_021.png)

**Figure 28** shows sampling results after 10 epochs for FasterDiT, with $v$-direction loss, *log-norm t* tricks.

![](assets_doc/doc_img_023.png)

**Figure 29** shows sampling results after 10 epochs for FasterDiT, with $v$-direction loss, *log-norm t* and standard deviation scaling (with target 0.82) tricks.

We found that the model can be sensitive to input standard deviation scaling. If the scaling target is not properly specified, e.g., 0.82 as default value suggested by paper, it actually made results slightly worse – likely because the scaling was not theoretically justified and thus “mis-trained” the model. In summary, **our flow model was quite robust to these variations and didn’t show obvious gains from the $v$-direction loss, log-norm time or standard deviation rescaling**. It’s possible that on more complex datasets or with larger models, these tricks could matter, but at least on MNIST their effect was minor. 

### Shortcut Model

We explored the idea of **“shortcut” generation via self-consistency** in the flow model, as in [Shortcut models](https://arxiv.org/abs/2410.12557). One appealing aspect of flow models is the potential to generate samples with *very few sampling steps* (even one step, in theory) if the learned continuous flow is simple enough. Recent research on *Consistency Models* (Song et al., 2023) has shown that it’s possible to train a model to jump directly from pure noise to data in one step by enforcing a self-consistency loss, or to achieve good results in 2–4 steps via a hybrid approach ([Multistep Consistency Models](https://arxiv.org/abs/2403.06807)). We tried to incorporate a **self-consistency objective** into our flow matching training: essentially, during training we occasionally “bootstrap” the model by sampling an intermediate state and forcing the model to map it to the true data distribution in one step (in addition to matching the continuous flow). In practice, we followed a strategy of using ~1/8 of the training batches for this self-consistency loss (and the rest for the usual flow matching) – this was inspired by techniques in consistency model training. After training the flow model with this combined objective (flow matching + self-consistency) for 20 epochs, we evaluated its ability to generate images in **fewer steps**. 

* **Inference steps:** We tried generation with the standard 128 integration steps (same 128 steps used in training), as well as truncated to 4 steps, 2 steps, and even 1 step for inference (training still 128 steps). The results showed that **the flow matching part was learned well (128-step generation was excellent), but the self-consistency part was not strong enough** – the 1-step and 2-step generations were still very poor (essentially indistinguishable noise or highly blurry digits), and 4-step was slightly better but nowhere near the quality of 128 steps. 

![](assets_doc/doc_img_027.png)

**Figure 30** shows sampling results for Shortcut model with 1-step inference, after training 20 epochs.

![](assets_doc/doc_img_028.png)

**Figure 31** shows sampling results for Shortcut model with 2-step inference, after training 20 epochs.

![](assets_doc/doc_img_029.png)

**Figure 32** shows sampling results for Shortcut model with 4-step inference, after training 20 epochs.

![](assets_doc/doc_img_030.png)

**Figure 33** shows sampling results for Shortcut model with 128-step inference, after training 20 epochs.



* **Training epochs:** Even after extending training to 200 epochs (10× longer), the one-step generation remained unsatisfactory, improving only marginally. This highlights a crucial lesson: **achieving high-quality one-step generation is very challenging**. Our attempt indicates that simply adding a small fraction of consistency loss is insufficient for the model to master the extremely nonlinear mapping from pure noise to data in one go. This is in line with observations by consistency model researchers – Jonathan Heek et al. (2024) noted that consistency models (single-step models) are much harder to train than diffusion models ([Multistep Consistency Models](https://arxiv.org/abs/2403.06807#:~:text=,model%20is%20a%20diffusion%20model)), and they proposed *Multistep Consistency Models* as a compromise that allows a spectrum between one-step and many-step generation. Their results show that allowing 2 to 8 steps greatly eases training while still giving big speedups in sampling. Our experiment reinforces that finding: our flow model could generate decent images in 4 or more steps, but collapsed at 1–2 steps. It likely needs a dedicated training recipe (or distillation from a many-step teacher, as done in consistency distillation) to successfully generate in one step.

  ![](assets_doc/doc_img_031.png)

  **Figure 34** shows sampling results for Shortcut model with 1-step inference, after training 200 epochs.

  ![](assets_doc/doc_img_032.png)

  **Figure 35** shows sampling results for Shortcut model with 2-step inference, after training 200 epochs.

  ![](assets_doc/doc_img_033.png)

  **Figure 36** shows sampling results for Shortcut model with 4-step inference, after training 200 epochs.

  ![](assets_doc/doc_img_034.png)

  **Figure 37** shows sampling results for Shortcut model with 128-step inference, after training 200 epochs.

  

* **Class Conditioning Format:** Another parallel with the diffusion experiments was class conditioning. We applied the same class conditioning approach in the flow model (i.e. providing either a **one-hot class vector embedding or an integer label embedding**). The flow model’s behavior mirrored what we saw before in DiT and Diffusion U-Net: **one-hot conditioning was significantly better**. The Flow Matching Transformer trained with one-hot class input produced clearly distinguishable digits of each class, whereas using an embedded integer class ID led to confusion and poorer quality. This consistency reinforces the earlier point – no matter the generative training paradigm (diffusion or flow), giving the model a clean, explicit representation of the conditioning signal (like a one-hot vector) makes it much easier for the model to utilize that information. So for flow matching models, one should also prefer one-hot or similarly expressive conditioning inputs, especially for small-scale tasks.

  ![](assets_doc/doc_img_024.png)

  **Figure 38** shows sampling results for Shortcut model with integer class conditioning, after training 10 epochs.

![](assets_doc/doc_img_025.png)

​	**Figure 39** shows sampling results for Shortcut model with one-hot class conditioning, after training 10 	epochs.

![](assets_doc/doc_img_026.png)

​	**Figure 40** shows sampling results for Shortcut model with one-hot class conditioning, after training 10 epochs.



In summary, the flow matching experiments demonstrate the **potential of Flow Matching to reduce training time** (half the epochs for similar performance) and confirm that **conditioning and model design insights from diffusion still apply** (e.g. one-hot conditioning). However, they also show that **ultra-fast sampling (one-step)** remains an open challenge – our naive approach wasn’t enough, echoing the literature that specialized techniques are needed for consistency models. Given that flow matching provides a flexible framework (we can choose any path), a promising direction for future work is to explore paths or objectives that are explicitly designed to yield simple flows that are easier to approximate in one step. Additionally, combining flow matching with multistep consistency training (as suggested by Heek et al.) could strike a good balance in practice.

## Tokenizer Design for Latent Diffusion

Finally, we turn to the **design of the tokenizer** in latent diffusion models. In image diffusion (especially high-resolution), it’s common to compress the image into a lower-dimensional latent space (e.g. via a VAE or a conv encoder) and then operate the diffusion process in that latent space. The tokenizer (encoder) and detokenizer (decoder) thus play a critical role: they determine the representation that the diffusion model works with. We performed experiments to understand how the *dimension of the latent tokens* affects training and generation. Instead of a pretrained VAE, we used simple learned convolutional encoders (since our focus was on relative differences rather than absolute quality).

First, we tried a **single-layer 2D convolution as the tokenizer**. This convolution had kernel size and stride equal to a patch size (e.g. 2), effectively “patchifying” the image into non-overlapping patches and projecting them linearly to token embeddings (and a corresponding deconvolution to reconstruct). In theory, if the patch size is 2×2 on a 1-channel image, each patch has 4 pixels – so *4 numbers can perfectly represent that patch*. Thus, the **minimal token embedding dimension** needed to **losslessly encode** a 2×2 patch is 4. We indeed found that a token dimensionality of 4 was sufficient to reconstruct the images (the tokenizer+decoder could learn an identity mapping). However, when we trained a diffusion transformer on these tokens, the model with token dim = 4 learned rather slowly. If we increased the token embedding dimension (say to 8, 16, or higher), the diffusion model’s training **converged faster, even though ultimately it reached a similar final loss**. In other words, giving the model more degrees of freedom per token (beyond the bare minimum) made optimization easier, at least initially. This makes sense: a higher-dimensional token can capture redundant or rich features of the image patch, which the transformer can leverage in modeling; with an exactly minimal token (4 numbers representing 4 pixels), the model has to work harder to infer relationships because there’s no extra capacity to encode higher-level abstractions at the token level. The downside of larger token dimensions, however, is that the **tokenizer starts to over-represent the data**, potentially making generation harder. We noted that extremely high-dimensional tokens can lead to overfitting or poor generation quality if the diffusion model isn’t scaled up accordingly. This phenomenon – *improving reconstruction vs. harming generation* – was recently discussed by [Yao et al. (2025)](https://arxiv.org/pdf/2501.01423). They observed an **“optimization dilemma”** in latent diffusion: **increasing the per-token feature dimension improves reconstructions of the input, but it requires a substantially larger diffusion model and more training to achieve comparable generation performance**. Otherwise, generation quality drops with high token dimension. Our findings are in line with this: a token dim of 4 (minimal) gave perfect reconstructions but training was slow; a very large token dim would reconstruct easily but make it hard for the diffusion model to generalize (since the latent space becomes high-dimensional and unconstrained). Thus, there’s a sweet spot where the token dimension is *slightly above the theoretical minimum* – providing some redundancy for easier modeling – but not too large to overwhelm the diffusion model. In our case, token dims in the range of 8–16 seemed to work well for the small data.

![](assets_doc/doc_img_035.png)

​	**Figure 41** shows training curves for 1D convolutional tokenizer (equivalent patch size 2x2) with different token embedding dimensions, by MINST image encoding-decoding loss.

We extended this experiment to a **2-layer convolutional tokenizer** (applying two convs with kernel size=stride=2, effectively a 4×4 patch overall). In that scenario, a 4×4 patch from the image has 16 pixels, so the minimal token vector length after two layers of downsampling would be 16. We found a very similar pattern: **16-dimensional tokens were sufficient but slow to train**, while larger token dims (e.g. 32 or 64) learned faster initially. Again, all eventually converged to comparable reconstruction fidelity, but the diffusion model trained on higher-dim tokens tended to reach lower loss quicker. The general insight across these tests is that **token embedding dimensionality is a trade-off**: enough dimensions to make learning easy, but not so many that the latent space is unnecessarily large for the generative model to handle.

![](assets_doc/doc_img_036.png)

​	**Figure 42** shows training curves for 2D convolutional tokenizer (equivalent patch size 4x4) and different token embedding dimensions, by MINST image encoding-decoding loss.

Now we consider the usage of such token encoder with DiT. One practical complication we encountered is that in the **DiT architecture, the token embedding dimension is tied to the model’s hidden dimension**. By default, if we choose a Transformer hidden size $d$, the token vectors (and time embedding) are of the same size $d$. In our experiment, this meant if we wanted token dim = 4, the whole model’s hidden size would be 4 – which is far too small for the Transformer to function well (it wouldn’t even satisfy multi-head attention requirements). In the actual DiT implementation, the minimum hidden size is constrained by things like number of attention heads and the use of RoPE. For example, with 8 attention heads of dimension 32 each, the hidden size must be at least $8 \times 32  \times 2 = 512$, and the last $2$ is because RoPE encodes features in pairs, the dimension should be even and ideally a multiple of 2 per head ([A Deep Dive into Rotary Positional Embeddings (RoPE): Theory and Implementation](https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32)). The DiT paper indeed set a relatively large hidden dimension partly for these reasons. 

To experiment with small token dims, we decoupled the tokenizer dimension from the Transformer’s internal dimension – essentially projecting a small token (e.g. 4-dim) up to a larger hidden size inside the model when calculating the Q, K and V matrices. This allowed us to test, say, token  embedding dim = 4 with a Transformer hidden size of 256 or 512 (so attention and time embeddings still operate in a reasonable space). Theoretically, for patch size 2x2, the token embedding dimension 4 should be sufficient to allow lossless encoding-decoding, as evidenced by previous experiments. However, we discovered that when using this approach, **in DiT training, very small token dims (even if theoretically sufficient to support lossless encoding) still resulted in poor diffusion model performance**, even if the model’s internal hidden size was moderate (e.g., 256 or 512). For instance, using 4-dimensional tokens projected into a 512-dimensional hidden space did not train as well as using 16-dimensional tokens projected into the same 512-dimensional hidden space. This suggests that **if the token representation itself is too low-dimensional, the model struggles**, even if you map it to a larger vector for processing Q, K and V later. Likely, the initial projection from token to hidden cannot create useful features if the token carried too little information to begin with. 

Moreover, **reducing token dimension provides only a limited computational gain overall** – it saves some work in the token projection layers, but the bulk of computation is in the Transformer’s attention and feed-forward layers, which still run in the higher hidden dimension. So, the benefit of making token dim extremely small is modest, yet it can hurt generative performance disproportionately. 

![](assets_doc/doc_img_037.png)

**Figure 43** shows training curves for DiT (patch size 2x2) with different token embedding dimensions (*hidden_size* in figure), like 16, 128 and 256, for MNIST image generation. Theoretically, for patch size 2x2 the token embedding dimension 4 should be sufficient to allow loseless encoding, but practical DiT training shows that even dimension 16 does not learn well. Theoretically the three curves could converge to the same optimal loss values, but are almost impossible in practice. The larger the models, the more significant difference of training efficiency is observed by varying token embedding dimensions.

Overall, the lessons from these tokenizer experiments are: **(a)** Even *theoretically sufficient* token dimensions might be suboptimal for training speed and generative quality – giving the model a bit more latent capacity than the bare minimum helps. **(b)** However, making the token dimension too large leads to diminishing returns and can degrade generation unless the model size and data scale are increased to compensate ([Reconstruction vs. Generation](https://arxiv.org/pdf/2501.01423)). And **(c)** Architectural constraints (like positional encoding requirements and coupling of dimensions) mean we often can’t shrink the latent as much as theory suggests without breaking assumptions. A practical approach to mitigate the high-dimensional latent issue is to use a *pretrained compression* – e.g., Yao et al. introduced a VAE aligned with a pretrained vision model to produce a latent space that is easier for the diffusion model to learn. Their **LightningDiT** leveraged this to successfully train with high-dimensional latents but still converge fast. In our case, without such alignment, we found a moderate latent dimensionality worked best for a given model size. 

To put it succinctly: **choose the token latent dimensionality wisely**. It should be large enough that the diffusion model isn’t information-starved and can learn quickly, but not so large that the generative modeling problem becomes harder than necessary. If one needs high-dimensional latents (for fidelity), then the diffusion model and training strategy might need to be scaled up to handle it – or use techniques to constrain the latent space (like adding perceptual losses or using pretrained encoders to guide it). This is an active area of research, as the trade-off between *reconstruction* and *generation* quality in latent diffusion is crucial for building models that are both efficient and effective.

## Conclusion and Future Directions

Across these experiments in Transformers, diffusion models, flow matching, and tokenizer design, we gained a number of insights that can inform future research:

- **Infinite Context vs. Optimization Speed:** Adding mechanisms like Infini-Attention can extend a Transformer's context window indefinitely, but **this comes at a cost to training convergence** – models may train slower or require more careful tuning to reach the same performance as standard Transformers. Future work might focus on improving the optimizability of such long-context models (better initialization, adaptive learning rates, etc.), since the ability to handle unbounded context is highly desirable.

- **Normalization Alternatives in Transformers:** We found that **normalization-free Transformers using Dynamic Tanh can perform on par with standard normalized Transformers**, confirming recent results ([Transformers without Normalization](https://arxiv.org/html/2503.10622v1#:~:text=Normalization%20layers%20are%20ubiquitous%20in,We%20validate%20the)). This suggests that some architectures could drop LayerNorm/RMSNorm layers without loss of quality, simplifying design and potentially improving speed. It will be interesting to apply this to large-scale models – e.g. will eliminating normalization affect very deep or large Transformers’ stability? Early evidence is positive, and removing norms could also avoid certain normalization-induced limitations.

- **Diffusion Model Design:** When designing diffusion models, **using a sufficient number of timesteps is crucial** for quality – too few steps severely hurt performance. Techniques like training with hundreds or thousands of steps and then using efficient samplers at inference are well-founded. **Input representation matters**: in Vision Transformers or other backbones, one must ensure the model isn’t losing detail (e.g. choose patch size carefully, consider hybrid CNN-token approaches for images). **Stronger conditioning and guidance** can dramatically improve sample fidelity in conditional diffusion – simple tricks like one-hot encoding labels or employing classifier-free guidance should be standard practice for small data scenarios. And as expected, **bigger models (more layers) yield better results**, so allocate capacity according to the complexity of data. The DiT vs U-Net comparison also indicates that while Transformers can be viable diffusion backbones, **specialized convolutional architectures still have an edge on image tasks** – an open question is how to combine or enhance Transformers to close this gap (perhaps using patches plus local convolution within each patch, or other inductive biases).

- **Flow Matching vs Diffusion:** Flow Matching is a promising avenue to train generative models more efficiently. Our experiments show **FM can reach target performance with significantly fewer training epochs** than an equivalent diffusion model, echoing the idea that one can bypass the inefficiencies of simulating every diffusion step during training ([Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)). For practitioners, FM could mean faster turnaround in training generative models. However, the challenge is that **FM (and related consistency models) aim for very fast sampling**, which proved difficult to fully achieve in our test – one-step generation did not match the multi-step quality. Recent research suggests using a small number of refinement steps (e.g. 4 or 8) is a sweet spot ([Multistep Consistency Models](https://arxiv.org/abs/2403.06807)). So, a **multi-step consistency training approach** might be the practical compromise: you still train faster than diffusion and sample in a handful of steps, without requiring the model to do everything in one step. We expect to see more work integrating flow matching with consistency distillation, and exploring different probability paths to make the learned flows as simple as possible for a given data distribution. 

- **Tokenizer and Latent Space Design:** In latent diffusion, **the choice of tokenizer (compressor) and its dimensionality has profound effects** on training and generation. Our findings reinforce an important trade-off: *lower-dimensional latents* make the generative modeling easier to learn (and require smaller diffusion models), but if taken too far, they can bottleneck the information and harm quality; *higher-dimensional latents* preserve detail and can improve reconstructions, as well as converging faster with same model architecture, but they demand more from the diffusion model and training process ([Reconstruction vs. Generation](https://arxiv.org/pdf/2501.01423). Going forward, **alignment of latent spaces with powerful pretrained features** (e.g. CLIP or DINO features for images) could resolve the reconstruction-generation dilemma, allowing us to have rich latents without needing exorbitant training regimes. Additionally, decoupling the token dimension from the model dimension (as we tried) is useful, but one must ensure architectural components (like positional encodings) are adjusted accordingly. The takeaway for practitioners is to **tune the token dimensionality** and consider techniques like perceptual loss or latent alignment if pushing to extreme settings. There is likely not a one-size-fits-all: the optimal latent size depends on the data complexity and the capacity of the generative model.

In conclusion, these experiments provided a deeper understanding of several cutting-edge modeling techniques. For **Transformers**, we see that new methods (infinite attention, norm-free layers) show promise but also come with new challenges. In **diffusion models**, paying attention to the “small things” – time steps, patches, conditioning – can significantly impact performance, and Transformers are emerging as an alternative backbone with their own considerations. **Flow matching** offers a new lens to look at generative training, potentially leading to faster and more flexible models if its hurdles (like one-step generation) can be overcome. And **tokenizer design** reminds us that the interface between data and model (the representation) is critical – it can either accelerate or hinder learning and generation. 

Each of these areas could be a post (or research paper) in itself, but by examining them side by side, we also realize how interconnected they are. For instance, a better tokenizer could help both diffusion and flow models; a faster training method like flow matching could benefit from the infinite context of new Transformers (imagine modeling very long videos or sequences directly); and so on. As the field progresses, we expect many of these ideas to converge – e.g., future generative models might use *infinite-context Transformers trained with flow matching on aligned latent spaces*. The possibilities are exciting. We hope the insights shared here help others navigate these design choices and inspire further experimentation to push the boundaries of generative modeling and sequence modeling. 

## Citation

```
@article{ding2025nuts,
  title   = "Nuts and Bolts in Training Transformers and Diffusion Models",
  author  = "Ding, Zihan",
  journal = "quantumiracle.github.io",
  year    = "2025",
  month   = "Apr",
  url     = "https://quantumiracle.github.io/webpage/blogs/blog20250415.html"
}
```

