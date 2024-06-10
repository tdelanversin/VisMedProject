# Exploring Komolgorov-Arnold Network applications in image classification tasks

Tilman de Lanversin - tilmand@g.ecc.u-tokyo.ac.jp  
ETH MSc student in Computer Science, Major in Visual and Interactive Computing  
Exchange student at the University of Tokyo in Creative Informatics

## Instructions:

- Read one paper presented in 2020-2024 at TOP journals/conferences 
- Implement them by yourself and submit source code to GitHub (make it open).
  - Authors’ original source code must not be used.
- Add an explanation explaining 
  - Why this paper is important (what the technical core is, why the paper is accepted)
  - What you have implemented

## Choice of model

As of early 2024, a new paper has been published to arXiv ([Liu Z. et al.][1]) with a revisiting of an old machine learning network model, the Komolgorov-Arnold Network. This architecture is presented as an alternative to the very commonly used MLP model, and is also a universal approximator.

The paper expands on previous ideas with modern techniques that work for MLPs, namely backpropagation to allow for arbitrarily deep networks to be used.  
This enables modern tasks to make use of KAN networks, with a few distinct benefits. As stated by the paper, the main advantages are a decreased parameter count, learnable non-linearities, increased interpretability.

While these were observed on simple testcases, and apply mostly to scientific calculations and are observed to be more useful mostly on deeper laten spaces ([GraphKAN][2]), this finding has a lot of potential.

## Choice of implementation

Taking into consideration these factors, I chose to try and implement the KAN as a step in a bigger image recognition pipeline. This would allow us to make use of it's strengths in later latent spaces, by first calculating all the convolutions before going into the KAN, as well as it's potential in explanability to show image features that the model is looking for.

I then pitched this idea to professor Yamasaki, who approved of it on slack here:

> **Tilman** *12:56*  
>> Hello professor Yamasaki, I'm contacting you with regards to the project for Visual Media. Would you consider implementing a ResNET architecture with a Komolgorov Arnold Network instead of an MLP a sufficient task for the project? I don't believe such a paper has been created, but I think the idea could be interesting to explore in a simple course project like this one. Thank you for your opinion
> 
> **Toshihiko Yamasaki (教職員)** *13:20*  
>> I do not know whether it has never been tried before, but it sounds interesting.
> When submitting your report, please paste our conversation or mention that I said yes.
> 
> **Tilman** *13:20*  
>> Thank you

## Exploring KANs

To start off, for me to have a good basis to compare the rest of my work with, I donwloaded a recently created KAN based model by Github user `GistNoesis` ([FourierKAN][3]) that uses fourier series instead of B-splines for the learnable function. This is also in a pytorch layer format, and will thusly allow me to very easily use it as a first implementation. 


---

[1]: <https://arxiv.org/abs/2404.19756> "Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y. and Tegmark, M., 2024. Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756."

[2]: <https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks?tab=readme-ov-file> "GraphKAN -- Implementation of Graph Neural Network version of Kolmogorov Arnold Networks (GraphKAN)"

[3]: <https://github.com/GistNoesis/FourierKAN?tab=readme-ov-file> "FourierKAN -- Pytorch Layer for FourierKAN"