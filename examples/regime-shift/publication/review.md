# 金融市场 regime shift：经济状态、统计状态、实时识别、样本外预测与配置价值：文献综述

## 引言

围绕“金融市场 regime shift：经济状态、统计状态、实时识别、样本外预测与配置价值”的研究并非单一方法或结论的累积，而是由概念边界、解释机制、经验识别和适用条件相互牵引的问题链。已有工作一方面扩展了可观察对象和分析工具，另一方面也不断暴露不同数据、情境与评价尺度之间难以直接通约的张力[1, 2, 5, 24, 8]。因此，真正需要解释的不是哪一种方法占据优势，而是各类证据在什么问题上形成共识、在什么条件下发生分歧，以及这些分歧如何推动新的研究问题。

现有讨论可沿若干彼此承接但并不必然线性的主题展开：基础性研究首先界定对象与核心争议，方法研究随后把抽象问题转化为可识别的经验命题，应用研究检验这些命题在不同场景中的稳定性，而对局限和边界的讨论又反过来修正最初的概念与方法假设。

## 可识别经验命题的构造

现有研究如何把核心问题转化为数据、模型与可执行的分析流程？现有研究的共同起点是：该组文献提出或比较了数据来源、建模方法与技术流程，扩展了问题的可测量性。[1, 2, 3, 4, 5, 6]。这些工作之所以构成同一讨论，并非因为结论完全一致，而是因为它们都从题名或摘要重点描述数据、模型、算法、测量工具或技术框架切入同一问题。

在解释路径上，相关研究主要使用This study proposes a regime-switching model driven by deep learning.；By integrating a bidirectional long short-term memory network with the probabilistic inference framework of the Markov transformation process, a unified optimization framework for…；<jats:p>The study used the Markov regime switching model to investigate the presence of regimes in the volatility dynamics of the returns of JSE All-Share Index (ALSI).；以及Estimates from the regime switching model were compared to the industry standard non-switching GARCH (1,1) using the Deviance Information Criteria (DIC).。这些方法提供了互补视角：一部分工作强调识别与测量，另一部分工作更关注结果在具体环境中的表现。综合可见，较为稳定的发现包括Experiments show that the model achieves significant performance advantages over traditional methods in structural break detection, mechanism transition identification, and volati…；Volatility regimes are as a result of sudden changes in the underlying economy generating the market returns.；The results show that the two-regime switching EGARCH model with skewed Student t innovations describes better the return of the JSE Index.；以及摘要未明确给出可可靠抽取的主要发现[7, 8, 9, 10, 11, 12]。然而，方法之间的差异并不只是技术选择，它们往往对应不同的研究对象、时间尺度和有效性标准，因而相似的结果未必具有相同含义，相反的结果也未必构成直接否定。

争议进一步集中在摘要未报告作者明确陈述的局限；Cryptocurrency markets are characterized by high volatility, structural breaks, and non-stationary behavior, which often limit the effectiveness of traditional linear time-series…；<jats:p xml:lang="fr"><p>The detection of change points in chaotic and non-stationary time series presents a critical challenge for numerous practical applications, particularly i…；以及The work highlights the potential for future advancements in neural network applications and multi-expert decision systems, further enhancing predictive accuracy in volatile envir…。方法能力本身并不能保证在真实场景中产生稳定且可迁移的效果。[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]。这使讨论自然转向一个更严格的问题：这些方法在实际应用和经验研究中表现如何，证据是否一致？

## 应用情境中的效应异质性与条件依赖

不同应用场景中的经验证据如何支持或修正方法层面的预期？现有研究的共同起点是：该组研究把方法置于具体对象与场景中检验，展示了效果的异质性和条件依赖。[2, 5, 9, 14, 19]。这些工作之所以构成同一讨论，并非因为结论完全一致，而是因为它们都从题名或摘要以应用案例、经验比较、效果评估或具体场景为中心切入同一问题。

在解释路径上，相关研究主要使用<jats:p>The study used the Markov regime switching model to investigate the presence of regimes in the volatility dynamics of the returns of JSE All-Share Index (ALSI).；Estimates from the regime switching model were compared to the industry standard non-switching GARCH (1,1) using the Deviance Information Criteria (DIC).；Hence, this study aims to model the Nigerian exchange rate volatility using the Markov regime-switching model.；以及The study analyses the Nigerian exchange rate returns in two and three distinct regimes by employing the Markov regime-switching autoregressive (MS-AR) model with data from 2nd Ja…。这些方法提供了互补视角：一部分工作强调识别与测量，另一部分工作更关注结果在具体环境中的表现。综合可见，较为稳定的发现包括Volatility regimes are as a result of sudden changes in the underlying economy generating the market returns.；The results show that the two-regime switching EGARCH model with skewed Student t innovations describes better the return of the JSE Index.；imbalance and liquidation regimes’ identification and their average durations, show that the Naira in the foreign exchange market is not favourable for investors to trade.；以及摘要未明确给出可可靠抽取的主要发现[2, 5, 9, 14, 19]。然而，方法之间的差异并不只是技术选择，它们往往对应不同的研究对象、时间尺度和有效性标准，因而相似的结果未必具有相同含义，相反的结果也未必构成直接否定。

争议进一步集中在摘要未报告作者明确陈述的局限；以及The deep financial turmoil in China caused by the COVID-19 pandemic has exacerbated fiscal shocks and soaring public debt levels, which raises concerns about the stability and sus…。场景化结果的差异说明，平均效果不足以界定方法的适用边界与外部有效性。[5, 9, 14, 19]。这使讨论自然转向一个更严格的问题：哪些风险、偏差与评价缺口限制了这些结果的推广？

## 证据边界、评价有效性与跨情境推广

现有证据在哪些风险、偏差、评价设计和外部有效性方面仍受限制？现有研究的共同起点是：该组文献揭示了技术与应用证据的边界，并把研究议程推进到治理、比较评价和可靠性问题。[24, 8, 9, 13]。这些工作之所以构成同一讨论，并非因为结论完全一致，而是因为它们都从题名或摘要明确讨论风险、伦理、治理、局限、挑战或评价有效性切入同一问题。

在解释路径上，相关研究主要使用摘要未明确给出可可靠抽取的方法细节；The research employs a multivariate Hidden Markov Model (HMM) with improved data preprocessing techniques and principal component analysis (PCA) to process data from 2010 to 2025.；以及The model shows a 100% probability that the market operates under Regime 8 which produces stable returns with a 1.6 Sharpe ratio during November 2025 while showing limited market…。这些方法提供了互补视角：一部分工作强调识别与测量，另一部分工作更关注结果在具体环境中的表现。综合可见，较为稳定的发现包括摘要未明确给出可可靠抽取的主要发现[24, 8, 9, 13]。然而，方法之间的差异并不只是技术选择，它们往往对应不同的研究对象、时间尺度和有效性标准，因而相似的结果未必具有相同含义，相反的结果也未必构成直接否定。

争议进一步集中在摘要未报告作者明确陈述的局限。现有研究仍缺少跨方法、跨数据和跨场景的一致评价框架。[24, 8, 9, 13]。这使讨论自然转向一个更严格的问题：未来如何建立可复现、可比较且能反映真实边界条件的证据体系？

## 总结与展望

总体而言，关于“金融市场 regime shift：经济状态、统计状态、实时识别、样本外预测与配置价值”的知识积累已经从对象识别推进到机制解释、经验检验与边界辨析，但这种推进不是简单的线性替代。不同研究传统在概念、数据和评价标准上的差异，既造成结论分化，也构成相互校正的基础[23, 19, 13]。

后续研究的关键不在于继续增加彼此孤立的案例，而在于建立能够比较竞争解释的研究设计，报告负结果与条件依赖，扩展跨时期、跨区域和跨制度环境的外部验证，并明确区分预测改善、机制识别与因果解释。只有当这些证据边界被同时纳入讨论，该领域才能从方法上的局部成功走向可累积、可比较的知识体系。

## 参考文献

[1] Junyu Wang. (2026). Deep learning-driven regime switching models for capturing structural breaks and volatility clustering in financial time series. *Future Technology*. https://doi.org/10.55670/fpll.futech.5.2.25
[2] Emmanuel K. Oseifuah, Carl H. Korkpoe. (2019). A Markov regime switching approach to estimating the volatility of Johannesburg Stock Exchange (JSE) returns. *Investment Management and Financial Innovations*. https://doi.org/10.21511/imfi.16(1).2019.17
[3] Sumanth Polavarapu. (2026). Cross-Asset Market Regime Detection Using Hidden Markov Models: A Framework for Real-Time Regime-Aware Risk Management. *Unknown venue*. https://doi.org/10.2139/ssrn.6539358
[4] Calandra A. Haryani. (2026). Market Regime Detection in Bitcoin Time Series Using K-Means Clustering and Hidden Markov Models. *Journal of Digital Market and Digital Currency*. https://doi.org/10.47738/jdmdc.v3i1.57
[5] Zira S.D., Adejumo O.A.. (2023). The Regime Examination of Nigeria Exchange Rate Volatility: Evidence from Markov Regime Switching Autoregressive Approach. *African Journal of Accounting and Financial Research*. https://doi.org/10.52589/ajafr-7mhoeggm
[6] Hossein Malekinezhad, Roya Rafati. (2026). Markov and Hidden Markov Models for Regime Detection in Cryptocurrency Markets: Evidence from Bitcoin (2024–2026). *Unknown venue*. https://doi.org/10.20944/preprints202603.0831.v1
[7] Tomoe Moore, Ping Wang. (2007). Volatility in stock returns for new EU member states: Markov regime switching model. *International Review of Financial Analysis*. https://doi.org/10.1016/j.irfa.2007.03.006
[8] José Pablo Dapena, Juan Andrés Serur, Julián Ricardo Siri. (2020). Risk On-Risk Off: A Regime Switching Model for Active Portfolio Management. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.3509895
[9] Zhi De Khoo. (2020). Multivariate Regime Switching GARCH Model Application to Portfolio and Risk Management. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.3580106
[10] Alexander Musaev, Dmitry Grigoriev, Maxim Kolosov. (2024). Adaptive algorithms for change point detection in financial time series. *AIMS Mathematics*. https://doi.org/10.3934/math.20241674
[11] Razvan Oprisor, Roy Kwon. (2020). Multi-Period Portfolio Optimization with Investor Views under Regime Switching. *Journal of Risk and Financial Management*. https://doi.org/10.3390/jrfm14010003
[12] Nguyet Nguyen, Dung T. Nguyen. (2020). Global Stock Selection with Hidden Markov Model. *Risks*. https://doi.org/10.3390/risks9010009
[13] Cemal Öztürk. (2025). Machine Learning-Driven Market Regime Analysis in Equity Markets: A Gaussian Hidden Markov Model Approach. *Modern Mikro İktisat: Teoriden Uygulamaya*. https://doi.org/10.58830/ozgur.pub1120.c4538
[14] Tianbao Zhou, Zhixin Liu, Yingying Xu. (2024). How do financial variables impact public debt growth in China? An empirical study based on Markov regime-switching model. *arXiv*. https://arxiv.org/pdf/2407.02183v1
[15] Hengzhu Liu, Ping Xiong, Dawei Jin, Lingzhi Yi. (2023). Change Point Detection and Trend Analysis for Financial Time Series. *2023 9th International Conference on Big Data Computing and Communications (BigCom)*. https://doi.org/10.1109/bigcom61073.2023.00026
[16] Zhebing Yan. (2025). Bayesian Change Point Detection in Financial Time Series: Evidence from the Hong Kong Stock Market. *Proceedings of the 2025 International Symposium on Machine Learning and Social Computing*. https://doi.org/10.1145/3778450.3778502
[17] Vivien Pei Wen Chua. (2026). Breadth Momentum and Hidden Markov Regime Detection: A Composite Signal for Tactical Equity Allocation. *Unknown venue*. https://doi.org/10.2139/ssrn.6965179
[18] Megang Nkamga Junile Staures, Audrius Kabašinskas. (2026). Identifiable Regime Detection in Pension Fund Networks via Sticky Hidden Markov Models. *Unknown venue*. https://doi.org/10.20944/preprints202606.0111.v1
[19] Kazeem Abimbola Sanusi, Zandri Dickason-Koekemoer. (2022). Cryptocurrency Returns, Cybercrime and Stock Market Volatility: GAS and Regime Switching Approaches. *International Journal of Economics and Financial Issues*. https://doi.org/10.32479/ijefi.13555
[20] Nicklas Werge. (2021). Predicting Risk-adjusted Returns using an Asset Independent Regime-switching Model. *arXiv*. https://arxiv.org/pdf/2107.05535v1
[21] Jay Salvi. (2025). Asymmetric Hidden Markov Modeling of Order Flow Imbalances for Microstructure-Aware Market Regime Detection. *Unknown venue*. https://doi.org/10.2139/ssrn.5315733
[22] Irma Palupi, Bambang Ari Wahyudi, Agung Perdana Putra. (2021). Implementation of Hidden Markov Model (HMM) to Predict Financial Market Regime. *2021 9th International Conference on Information and Communication Technology (ICoICT)*. https://doi.org/10.1109/icoict52021.2021.9527459
[23] Xiaoyue Li, John M. Mulvey. (2021). Portfolio Optimization Under Regime Switching and Transaction Costs: Combining Neural Networks and Dynamic Programs. *INFORMS Journal on Optimization*. https://doi.org/10.1287/ijoo.2021.0053
[24] Andreas Reuss, Pablo Olivares, Luis Seco, Rudi Zagst. (2016). Risk management and portfolio selection using \alpha-stable regime switching models. *Applied Mathematical Sciences*. https://doi.org/10.12988/ams.2016.512722
