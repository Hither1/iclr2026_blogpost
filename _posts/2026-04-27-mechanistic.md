---
layout: distill
title: A Mechanistic Analysis of Low-Precision Instabilities in Microscaling Formats
description: Training large language models is expensive and compute-bound, and it must be repeated as models scale, algorithms improve, and new data is collected. To address this, next-generation hardware accelerators like NVIDIA’s Blackwell increasingly support lower-precision arithmetic formats, including Microscaling (MX) formats. In this work, we investigate the challenges and viability of block-scaled precision formats during model training. Across a broad sweep of weight-activation precision combinations and compute budgets from ( 2 \times 10^{17} ) to ( 4.8 \times 10^{19} ) FLOPs, we generally observe that training in MX formats exhibits sharp, stochastic instabilities in the loss, particularly at larger compute scales. To explain this phenomenon, we conduct controlled experiments and ablations on a smaller proxy model that exhibits instability behavior similar to the language model, sweeping across architectural settings, hyperparameters, and precision formats. These experiments motivate a simple model in which multiplicative gradient bias introduced by the quantization of layer-norm affine parameters and a small fraction of activations can trigger runaway divergence. Through \textit{in situ} intervention experiments on our proxy model, we demonstrate that instabilities can be averted or delayed by modifying precision schemes mid-training. Guided by these findings, we evaluate stabilization strategies in the LLM setting and show that certain hybrid configurations recover performance competitive with full-precision training.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: 
#     url: ""
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: ""
#     affiliations:
#       name: IAS, Princeton
#   - name: 
#     url: ""
#     affiliations:
#       name: IAS, Princeton
#   - name: 
#     url: ""
#     affiliations:
#       name: IAS, Princeton
#   - name: 
#     url: ""
#     affiliations:
#       name: 


# must be the exact same name as your blogpost
bibliography: 2026-04-27-mechanistic.bib


#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 
  - name: 
    subsections:
      - name: 
  - name: 
  - name: 
  - name: 
  - name: 
  - name: 
  - name: 

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## Introduction
Large language models (LLMs) have improved dramatically in recent years, largely by scaling their capacity and the quantity of training data <d-cite key="kaplan2020scaling, openai2025gpt45, deepmind2025gemini25, anthropic2025claude4, grattafiori2024llama"></d-cite>. For instance, training the Llama 3.1 405B model required more than 1025 FLOPs and utilized up to 16,000 H100 GPUs <d-cite key="grattafiori2024llama"></d-cite>. Scaling these models involves not only the initial, compute-intensive pretraining
phase but also frequent retraining as new data, algorithms, or architectures emerge, as well
as post-training protocols that prepare the model for inference/deployment.

To reduce these computational burdens, recent hardware advancements have introduced
native support for lower-precision computations, such as FP8 training in NVIDIA H100
GPUs <d-cite key="micikevicius2022fp8formatsdeeplearning, noune20228bitnumericalformatsdeep"></d-cite>. Hardware accelerators powered by
NVIDIA’s Blackwell architecture further extend these capabilities with standardized, sharedscale Microscaling (MX) formats like MXFP8 and MXFP6 (NVIDIA, 2025). These formats
store a per-block shared scale, which expands the effective dynamic range with minimal
memory overhead, while simultaneously enabling GEMMs at lower precision (Rouhani et al.,
2023; Darvish Rouhani et al., 2023b). While pretraining is typically done in 16 or 32-bit
precision, some quantization schemes are already seeing industry adoption; for example,
DeepSeek-V3 employs tile-wise FP8 quantization within large tensors <d-cite key="liu2024deepseek"></d-cite>, while
Cohere’s Command A model was trained in FP8 while reserving higher-precision operations
for activation functions and attention mechanisms <d-cite key="cohere2025commandaenterprisereadylarge"></d-cite>. At an even larger scale, the Llama-4 series of models is reported to have been pretrained in FP8 precision
across nearly 32,000 GPUs <d-cite key="llama4"></d-cite>. On the deployment side, methods like QAT and
mixed-precision fine-tuning further underscore the importance of understanding low-precision
training dynamics <d-cite key="jacob2017quantizationtrainingneuralnetworks, Abdolrashidi_2021, shao2024omniquantomnidirectionallycalibratedquantization"></d-cite>.

Two primary challenges accompany the adoption of low-precision formats for training. First,
there is a potential performance tradeoff, where reducing precision may result in degradation
of loss and downstream accuracy, which can be characterized through scaling laws that account
for both compute and precision <d-cite key="kumar2024scaling"></d-cite>. Second, instabilities during training can
occur, often manifesting as abrupt spikes in the loss curve that disrupt convergence <d-cite key="fishman2024scaling, lee2025fp8againquantifyingreduced"></d-cite>. When these instabilities push optimization into regions from
which recovery is impossible, they obstruct our ability to extract valid scaling laws, making
it impossible to even assess the tradeoffs introduced by low-precision training.

In this work, we set out to understand the training dynamics of low-precision MX precision
formats to identify format prescriptions for language model training on next-generation
hardware. However, like prior observations on (albeit non-MX) low-precision training
by <d-cite key="fishman2024scaling, lee2025fp8againquantifyingreduced"></d-cite>, we found that training frequently became
unstable, particularly for larger, compute-intensive models. The instabilities are pervasive,
emerging across a broad range of activation functions, model scales, quantization formats,
and hyperparameter settings.

Because large-scale language model (LM) sweeps are computationally intensive and involve
many entangled components, we turn to a controlled synthetic setting to understand the
origin of these instabilities. Specifically, we present a residual multi-layer perceptron (MLP)
model that captures key architectural components of the LM, and allows us to identify
conditions under which training becomes unstable. In particular, we are able to perform
hyperparamter sweeps, ablations across MX configurations, quantization schemes (e.g.,
forward-only vs. full quantization), and activation functions, and analyze their effects on
stability.

Our findings support a phenomenological explanation in which training instabilities primarily
arise from systematic bias in gradient estimates introduced by quantization. We find that the
primary contribution to this bias is the quantization of the layer normalization (layernorm)
affine weights, whose values often become tightly clustered over the course of training. When
the values within a block converge too closely, division by the shared block scale can clamp
all values in that block to the largest representable number, destabilizing training. We verify
that this mechanism is not limited to synthetic settings but also emerges in the LM setting
by evaluating mitigation strategies to stabilize LM training, including disabling layernorm
quantization and using high precision in selective parts of the network computation.





<!-- $$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$ -->



## Related Work

### Low-Precision Instabilities
Training large Transformer models at scale can reveal instabilities that can disrupt or even
halt learning <d-cite key="liu2024deepseek, chowdhery2022palmscalinglanguagemodeling, dehghani2023scalingvisiontransformers22, zhang2022optopenpretrainedtransformer, molybog2023theoryadaminstabilitylargescale, fishman2024scaling, zoph2022stmoedesigningstabletransferable, ma2025understandingsilentdatacorruption, takase2025spikemorestabilizingpretraining"></d-cite>. In some cases, these issues are exacerbated or directly triggered by low-precision
quantization. For example, <d-cite key="fishman2024scaling"></d-cite> demonstrate that FP8 pretraining becomes
unstable when combined with the SwiGLU activation function, attributing the issue to an
outlier amplification effect that worsens due to progressive weight alignment over the course
of training. Similarly, <d-cite key="lee2025fp8againquantifyingreduced"></d-cite>  report that approximately 10% of BF16 runs using
the NanoGPT codebase fail to converge, whereas full-precision (TF32) training exhibits no
such failures. Other works <d-cite key="sun2024massive, bondarenko2023quantizabletransformersremovingoutliers,xuimproved"></d-cite>, point
to activation outliers and gradient norm growth as contributors to these failures while  tseng2025trainingllmsmxfp4 proposes a stochastic rounding based algorithm to stabilize training in MXFP4
formats. Meanwhile, DeepSeek-V3 also attributes certain training failures due to blockwise
quantization of activation gradients <d-cite key="liu2024deepseek"></d-cite>, underscoring the breadth of challenges
introduced by quantization schemes.  <d-cite key="DBLP:conf/iclr/WortsmanLXEAACG24"></d-cite> use small-scale proxy models to
study training instabilities in the context of growth of output and layer logits. We adopt a
similar approach, and use a simplified proxy model to understand the origin of low-precision
instabilities in LLMs.

### Review of MX Formats and Experimental Approach
MX formats are a class of low-precision numerical representations designed to enhance the
efficiency of deep learning models <d-cite key="ocp_mx, rouhani2023microscaling"></d-cite>.
We defer a detailed review of the MX scheme to Appendix A. To summarize, we represent
a block of k values, $$ {Vi}k_{i=1} $$, using a single shared scale factor X and k corresponding
low-precision elements {Pi} where the Pi are obtained by casting Vi/X to the specified
low-precision format. We present results for a block size k = 32 to match what will be
hardware supported. The scale X is calculated using $$ X = 2⌊log2 (maxi(|Vi|))⌋−emax elem $$ where
$$ emax elem $$ is the exponent of the largest normal number representable in the chosen element
data format.
In our experiments, we quantize both weights and activations using these MX formats using
the MX Pytorch Emulation Library <d-cite key="mx_library"></d-cite>. As described in Appendix A, this
quantization is applied dynamically to the inputs of matrix multiplication operations.

## LLM Experiments
### Setup
For our LM experiments, we use OLMo <d-cite key="groeneveld2024olmo"></d-cite> combined with the MX
PyTorch Emulation Library <d-cite key="mx_library"></d-cite> to enable training under various low-precision
configurations. All language models use the GeLU activation function; full hyperparameter
details are provided in Table 3. We sweep over a wide range of MX precision formats for both
weights and activations, including two FP6 variants (E3M2, E2M3), two FP8 variants (E4M3,
E5M2), and a bfloat16 baseline. Each configuration applies full quantization to both forward
and backward passes to both weights and activations, as implemented in the Microscaling
library <d-cite key="mx_library"></d-cite>. For each format, we train approximately 70 models1
spanning compute budgets from 2 × 1017 to 4 × 1019 FLOPs. Model sizes range from ∼20M to
∼1.7B parameters. Token counts are determined using an adapted version of the FLOP
accounting code from <d-cite key="brandfonbrener2024loss"></d-cite> Brandfonbrener et al. (2024), originally developed for OLMo scaling
law experiments. Token-to-parameter ratios in our sweep range from approximately 2 to
1.   Models are trained on the Fineweb-Edu dataset Penedo et al. (2024) <d-cite key="penedo2024the"></d-cite> and the StarCoder
dataset Li et al. (2023) <d-cite key="li2023starcoder"></d-cite>, with the longest runs trained on 35B tokens and the shortest runs
corresponding to models trained on 301M tokens.

### Instabilities in Low Precision
Figure 1a shows the training loss and gradient norm trajectories for bfloat16 models. Training
remains stable, with smooth convergence. By contrast, Figure 1b illustrates example
instabilities in the MXFP8 E5M2-E5M2 weights-activations configuration, where some training
runs exhibit sharp upward spikes in loss and large increases in gradient norm magnitude.
We find these instabilities to be common across other low-precision MX configurations and
hyperparameter settings, as documented in Appendix J. We observe the instabilities mainly
in larger, longer-trained models and that importantly, when training is destabilized, training
does not recover, and the loss continues to diverge. While the loss spikes appear abruptly,
the gradient norm typically grows more gradually (see, e.g., examples in Appendix J) and
fails to decrease over time as seen in stable bfloat16 training. This behavior strongly suggests
biased gradient estimates, a point that we will investigate further in subsequent sections.

## Synthetic Experiments

### Setup
Our LM experiments with OLMo involve many potentially interacting components, and it is
computationally expensive to determine exactly where the low-precision failure mode occurs.
To facilitate this task, following <d-cite key="DBLP:conf/iclr/WortsmanLXEAACG24"></d-cite>, we develop a small-scale proxy
model. Given an input $$ x ≡ A0 ∈ R dmodel $$, we consider a network composed of L residual
layers indexed by $$ k = 0, . . . , L − 1 $$. The hidden state at each layer is computed as:
$$
hk = W(1)
k LN(Ak−1), Ak = Ak−1 + W(2)
k
ϕ(hk), (1)
$$
where LN denotes layer normalization and ϕ is the activation function (e.g., ReLU, GeLU,
SwiGLU). Each residual block contains two weight matrices: $$ W(1)k $$
projects to the hidden dimension, and $$ W(2)k $$ projects back to dmodel. By default, the hidden size is set to $$ 4d_{model}^2 $$
This student/proxy model is only useful insofar as it (at least partially) mimics the failure
modes of the LM setting, so let us note the simplifications performed on the language
model in order to obtain the proxy model. First, we dispense with the self-attention blocks
since ablating over attention did not change the qualitiative nature of the divergences we
observed. Second, we remove the embedding layers since our goal is to understand exactly
how low-precision block scaled arithmetic biases gradient computations, as well as simplify
the various types of LM layernorms (such as QK-norms) into a single layernorm. Finally,
we also train with MSE loss rather than cross-entropy, although we experimented with a
distributional KL loss and again did not observe qualitative differences. While we show that
this model nevertheless remains instructive and predictive of the mechanistic origins of the
LM instabilities, we caution that stability in this minimal model as a necessary (though
perhaps not sufficient) condition for stability in the full LM. Appendix D inludes more
experiments on how some of these simplifications affect the training dynamics of the model.
The targets are generated by a fixed auxiliary/teacher model that serves as a sufficiently
complex learnable function <d-cite key="lin2025scalinglawslinearregression"></d-cite>, and whose architecture can be taken to be
the same as the student’s without the layer normalization. For sweeps where we change the
depth and width of the student, we similarly scale the teacher model. A small Gaussian label
noise (σ = 10−3) is added to the outputs. The inputs x are drawn i.i.d. from a standard
Gaussian, without cycling, using a fixed seed to ensure consistent batch order.


To isolate the effect of precision, we train two copies of the student model from the same
initialization. The first is trained in full precision (FP32). After training, the weights are
reset to their initial state and retrained using a low-precision MX format, with quantization
applied to both forward and backward passes as described in Section 2.2. Because the
random seed, kernel determinism, initialization, data, and batch order are identical, any
behavioral difference is attributable mainly to the change in precision.


Hyperparameter choices A key point explicated in Appendix C is that there are
hyperparameter choices for which the model in Equation (1) will give rise to train instabilities
(even in FP32 precision). This is not necessarily a precision issue, but rather due to the
fact that in any SGD method there exists some small probability of taking wrong gradient
step(s). If the size of the steps are large due to, e.g., a large learning rate, this will be
visible as a sudden spike(s) in the loss. In order to move away from these “expected”
instabilities, before ablating or changing various components of the architecture, we carefully
tune hyperparameters for each depth and width configuration in which all high-precision
runs are stable, but low precision is not (at least for a canonical choice of activation function
such as GeLU). For the same reason, we fix a moderately large batch size (2048) throughout
to reduce variance in gradient estimates.



### The Effect of Activation Functions and layernorms

Having fixed a hyperparameter regime in which instabilities only appear in low precision,
we first ablate the choice of activation function and the inclusion of layer normalization.
In Equation (1), this corresponds to varying $$ ϕ(·) $$ and including the presence of LN(·).

In Figure 2a, we observe that with layer normalization enabled, both GeLU and SwiGLU
activations exhibit instability in low precision, with SwiGLU being significantly more prone
to divergence. This is consistent with the findings of  <d-cite key="fishman2024scaling"></d-cite> Fishman et al. (2024), though our
results show that SwiGLU also destabilizes training in high precision, suggesting that it
generally increases stochasticity at least for this particular choice of hyperparameters, though
these instabilities are generally recoverable in high precision. We observe two irrecoverable
instabilities in GeLU under low precision that are absent in high precision.

Next, we look at the inclusion of layernorm. In Figure 2b, we observe that the loss improves
with the removal of layernorm. This is expected as the teacher network does not contain a
layernorm so that student model is able to more accurately represent its outputs. However,
removing layernorm tends to stabilize low-precision training runs and destabilize high
precision runs (for the same choice of hyperparameters in Figure 2a). At first glance, these
results are perplexing since it appears that low precision is more robust to removal of
layernorms. We will return to this point in Section 5 when we explicate the subtleties of
layernorms in block scaling formats.


## Overflow Dynamics

Typically, instabilities in low precision happen due to over/underflow issues that can bias the
gradient. However, in a block scaling format, it is unclear how gradient bias can accumulate
when the shared scale explicitly puts nearly all values within a representable range.


### Overflow Issues with layernorms

To understand this, we begin by examining a concrete example of MXFP8 E4M3 as specified
in <d-cite key="ocp_mx"></d-cite>. The left panel of Fig. 3 plots the relative gap $$ (xt+1−xt)/xt $$
between successive positive codes in this format, ordered from index 0 (the smallest subnormal, $$ 2−9 $$) up to index 125 (448). The index stops at 125, rather than the expected
2
7 − 1 = 127, because S 1111 1112 is reserved for the NaN symbol, which would otherwise
correspond to a value of 480, and S 0000 0002 is the zero code, leaving 126 remaining
codes <d-cite key="ocp_mx"></d-cite>. We can note the following:

1. For a fixed exponent bin the relative gap starts at 12.5% and decays to 6.6% as the
mantissa increases.
1. There is an overflow region (left of Figure 3) when the value exceed the  largest representable
normal number (448). Typically, these values are clamped down to 448.



### Potential Mitigations
To clearly establish causality of which components can (de)stabilize training, we ask whether
an impending divergence can be averted by in-situ interventions to the training recipe. 
Figure 4 tracks a configuration that is stable in FP32 but diverges in MXFP8 E4M3. This
setting corresponds to the previously described student-teacher scenario with four layers and
model dimension dmodel = 512. The instability starts approximately at step 5090 and we
consider interventions just before the instability, at step 5080, and well before the instability,
at step 4500. For each intervention we keep the random seed, model state, and batch sequence
identical, so the training state at the intervention step is the same as in the baseline run, so
any divergence afterward is therefore solely attributable to the intervention.


Key Takewaways The dominant MX precision-specific bias comes from overflow of
clustered layer-norm affine weights (and a small fraction of activations). Our intervention
experiments show that raising precision in key parts of the computation, such as increasing
the precision of layer norms or activations, can greatly improve stability.



##  Stabilization Strategies in LM Setting
Motivated by the effective mitigations observed in our synthetic experiments, we return
to the language-model (OLMo) setting and consider two training strategies: (1) retaining
bfloat16 as the element format for activations and layer norms, and (2) applying MX
quantization only to the forward pass. We emphasize that these are diagnostic and not
production-ready mitigations. Keeping activations in bfloat16 generally yields no computethroughput gain on hardware where the MMA executes in bfloat16, because mixed-operand
kernels typically upcast the lower-precision operand to the MMA precision. Conversely,
downcasting activations to low precision during the matmul would reintroduce the very
instabilities we aim to avoid. We defer a more fine-grained study of which layers truly require
high-precision activations to future work. Likewise, quantizing only the forward pass can at
most accelerate the forward fraction of training. Under standard assumptions, the backward
step costs roughly twice the forward, so the idealized wall-clock speedup is capped near
∼33%.

In both cases, we find that training remains stable across all FP8 configurations. Table 1
reports validation loss differences relative to full-bfloat16 baselines. MXFP8 E4M3 weights
paired with bfloat16 activations in particular match full-precision performance across all
tested model sizes. In Appendix G, we study how these results scale with compute and fit
valid Chinchilla-style scaling laws. Full loss curves and scaling law fits for both mitigation
strategies compared to bfloat16 baselines are also provided in Appendix G.



## Conclusion
We showed that training LLMs in shared-scale/MX configurations can lead to sharp, unrecoverable instabilities. Using large-scale LLM sweeps and a simple proxy model trained
on synthetic data, we isolate a failure mode of quantization-induced gradient bias, where
shared-scale clamping (particularly of layer-norm affine weights and to a lesser extent, other
activations) injects gradient noise that ultimately destabilizes training. We evaluated several
diagnostic mitigations, and found that stability can be preserved using higher precision in
selective parts of the network computation.
Looking ahead, continued hardware advances will expand the frontier of what is computationally feasible. Some concrete directions include: extending our proxy model to include
mixture-of-experts with many layers, and other transformer-specific components to better
predict instabilities; developing a clear theoretical picture of instabilities in optimization (see
Appendix B); and designing new blockwise scaling schemes such as in Mishra et al. (2025)
that adapt to skewed or tightly clustered distributions.





<!-- ```markdown
{% raw %}{% include figure.liquid path="assets/img/2026-04-27-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
``` -->


{% include figure.liquid path="assets/img/2026-04-27-distill-example/iclr.png" class="img-fluid" %}




<!-- <div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-distill-example/8.jpg" class="img-fluid z-depth-2" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-distill-example/10.jpg" class="img-fluid z-depth-2" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-distill-example/11.jpg" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path=".jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="" class="img-fluid" %}
    </div>
</div> -->




<!-- ```markdown
{% raw %}{% include [FIGURE_NAME].html %}{% endraw %}
```


```python
import pandas as pd
import plotly.express as px

fig = px.density_mapbox(
    df, lat='Latitude', lon='Longitude', z='Magnitude', radius=10,
    center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain")
fig.show()

fig.write_html('./assets/html/2026-04-27-distill-example/plotly_demo_1.html')
``` -->




{% raw %}
<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-distill-example/plotly_demo_1.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="600px"
    width="100%"
  ></iframe>
</div>
{% endraw %}




<div class="l-page">
  <iframe src="{{ 'assets/html/2026-04-27-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>





<d-footnote></d-footnote>






{% highlight c++ linenos %} <br/> coe <br/> {% endhighlight %}
{% endraw %}

{% highlight c++ %}


{% endhighlight %}

 [mermaid.js](https://mermaid-js.github.io/mermaid/){:target="\_blank"} directly.
 [](){:target="\_blank"} syntax.


```yaml
mermaid:
  enabled: true
  zoomable: true # optional, for zoomable diagrams
```



```mermaid
    participant Alice
    Alice->>John: Hello John, how are you?
```




<blockquote>
</blockquote>


 `d-article` 

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

 try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

 `.l-screen`.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

`.l-body`

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

---


 _asterisks_ (`*asterisks*`) or _underscores_ (`_underscores_`).

 **asterisks** or **underscores**.

 **asterisks and _underscores_**.



1. 
2. 

- 

1. 
   1. 
2. 
  

- 

* 

- 



<!-- [](https://www.google.com "Google's Homepage") -->


[][1]



[ text]: 
[1]: http://slashdot.org
[link text itself]:



![alt text]( "Logo Title Text 1")


![alt text][logo]



```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```



| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
|       |  |  |
|       |       |    |
|  |       |    $1 |




| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| _Still_  |  |  |
| 1        | 2         | 3          |


