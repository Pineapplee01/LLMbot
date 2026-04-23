# Research Wiki Activity Log

*Append-only timeline of all wiki mutations*

---

## 2026-04-15

**00:00** 鈥?LMBot RoBERTa鈫扲GT 鍩虹嚎澶嶇幇瀹屾垚 + Failure-Regime 璇婃柇 v2 瀹屾垚

**鍩虹嚎澶嶇幇**:
- 淇 `GNNs.py` 涓?RGT 鐨?`MLP_Model` 鏈畾涔?bug锛堟浛鎹负 `nn.Linear`锛?- 淇 `parser_args.py` 缂哄皯 `--dataset` 鍙傛暟
- 淇鏈嶅姟鍣?`torch_scatter` CPU 鐗堟湰锛堝畨瑁?CUDA 鐗堟湰锛?- 5 seed 鍏ㄩ儴瀹屾垚锛歊GT GNN F1=0.8619 (mean), RGCN GNN F1=0.8723 (mean)
- 鏂板: exp:lmbot_rgt_baseline

**璇婃柇 v2 鍏抽敭淇**:
1. 娑堥櫎 oracle contamination锛氬叏鍥?11826 鑺傜偣鍋氱湡瀹?GNN 鎺ㄧ悊
2. Regime 瀹氫箟鏀逛负绾墠楠屼俊鍙凤細`prop_corruption_score` 涓嶅啀浣跨敤 `gnn_correct`
3. 鏂板 matched-budget 椋庨櫓鎹曡幏姣旇緝锛圓UROC/AUPRC/top-k precision锛?4. 鏂板: exp:failure_regime_diagnostics_v2

**璇婃柇鍏抽敭鍙戠幇**:
- prop_corruption_flag锛堝墠楠屽畾涔夛級閿欒瀵岄泦 1.84x
- Disagreement top-5% precision 49.8%锛屼絾 AUROC 浠?0.576锛堥珮绮惧害浣庤鐩栵級
- entropy+disagree 缁勫悎鏈€浼橈紙AUROC 0.820, AUPRC 0.405锛?- Semantic ceiling 浠?2.7%锛圠M-only-correct锛夆€?璇箟澧炲己鏄獎鎿嶄綔绗?- both_wrong 12.1%锛屽瘜闆嗕簬 low-homophily 鍜?prop_corruption

**鏂板 claim**: C8 (Different Failure Regimes Prefer Different Actions) 鈥?PRELIMINARY
**鏇存柊 claims**: C1 (鏂板 v2 璇佹嵁), C4 (鏀规爣棰? 鏂板 matched-budget)
**鏇存柊 gap_map**: G1 (regime-specific rescue), G5 (matched-budget)
**鏂板 edges**: 5 鏉?
---

## 2026-04-10

**02:00** 鈥?rw5_repair 瀹炵幇瀹屾垚

**瀹炵幇鍐呭**:
- 鏂板鏂规硶锛歚dual_router_trainer.py::_apply_repair_integrated_fusion()` (lines 1140-1245)
- 闆嗘垚鍒拌矾鐢遍€昏緫锛歚dual_router_trainer.py` line 1026
- 鍛戒护琛岄€夐」锛歚main.py` 鏂板 `--rewrite_baseline_mode rw5_repair`

**鏍稿績鏈哄埗**:
1. 妫€娴?disagreement 鑺傜偣 (`pred_sem != pred_graph`)
2. 瀵?disagreement 鑺傜偣搴旂敤鍥句慨澶嶏紙浠呭壀鏋濓紝涓嶆坊鍔犺竟锛?3. 鍦ㄤ慨澶嶅悗鐨勫浘涓婇噸鏂颁紶鎾?4. 杞瀺鍚堬細`prob_final = 伪 * prob_sem + (1-伪) * prob_graph_repaired`

**璁捐鍐崇瓥**:
- **Prune-only**: Kill test B4 楠岃瘉浜嗕粎鍓灊瓒冲锛屾棤闇€娣诲姞杈?- **Disagreement trigger**: 浠呭湪涓撳涓嶄竴鑷存椂淇锛岃€岄潪鎵€鏈変綆缃俊搴﹁妭鐐?- **Re-train GNN**: 瀹屾暣閲嶆柊浼犳挱浠ヨ幏寰楀噯纭鐜?
**棰勬湡鎬ц兘**:
- Vanilla rw5: F1=0.85-0.88
- + Graph repair: +0.03-0.05 (鍒╃敤 rw4 鐨?+0.18 disagree gain)
- **Total**: F1=0.88-0.93

**鏂板鏂囨。**:
- `docs/wiki/ideas/004_rw5_variants.md` 鈥?璁板綍 3 涓?rw5 澧炲己鍙樹綋

**涓嬩竴姝?*:
1. 杩愯 rw5_repair 瀹為獙锛?-8 灏忔椂 GPU锛?2. 楠岃瘉 kill tests锛圔2: disagree-slice F1 gain 鈮?+0.18锛?3. 鍐崇瓥鐐癸細F1 鈮?0.88 鈫?multi-seed锛屽惁鍒欏疄鐜?rw5_learned

---

## 2026-04-09

**00:00** 鈥?鐮旂┒绛栫暐璁ㄨ涓庡喅绛栨洿鏂?
**鍐崇瓥1: 鐮旂┒鐩爣纭 (Option B)**
- 鐩爣锛氬湪 low-degree / graph-missing slice 涓婃彁鍗?3-5% F1
- 涓嶈拷姹?global SOTA锛坆enchmark 宸查ケ鍜岋級
- Novelty锛歀LM calibration uncertainty 鎸囧鍥句慨鏀癸紝鑰岄潪绠€鍗?LLM embedding 鏀瑰浘
- 鏂板锛歞ocs/wiki/project/research_strategy.md

**鍐崇瓥2: 鏁版嵁闆嗗垏鐗囦笉閲嶆柊鐢熸垚**
- 浣跨敤鐜版湁 `datasets/TwiBot-20/train_idx.pt` 绛夋枃浠讹紙canonical splits锛?- BotBR 宸茬‘璁や娇鐢ㄥ悓涓€濂?split锛坆otbr/Dataset.py:56-58锛?- Split 姣斾緥绾?70%/20%/10%锛屾€昏妭鐐?11826
- 鏂板锛歞ocs/wiki/project/dataset_split_decision.md

**鍐崇瓥3: HyperScan 閫傞厤寰呯‘璁?*
- HyperScan 鍘熷璁捐閽堝 MGTAB锛岄潪 TwiBot-20
- 闇€鍦ㄦ柊浼氳瘽涓‘璁ゆ槸鍚﹀凡閫傞厤锛屽惁鍒欐敼涓哄紩鐢ㄨ鏂囩粨鏋?
**鍐崇瓥4: Baseline 澶嶇幇璁″垝**
- LMbot + BotBR 浼樺厛澶嶇幇锛堜唬鐮佸吋瀹癸紝split 涓€鑷达級
- 闇€瀹炵幇缁熶竴 slice evaluation framework锛坧re-registered slice definitions锛?- 鏂板锛歞ocs/wiki/experiments/baseline_reproduction_plan.md

**鏇存柊鏂囦欢**: index.md, log.md, query_pack.md

**12:00** 鈥?澶ц妯℃洿鏂帮細鏁村悎 idea-discovery / run-experiment / experiment-bridge / novelty-check 浼氳瘽缁撴灉

**鏂板 experiments**:
- exp:kill_tests_b2_b3 鈥?鉁?BOTH PASSED (B2: repair +0.16 disagree F1, B3: ablations validated, B4: prune-only sufficient)
- exp:confidence_fallback 鈥?Fallback beats router +1.0% F1, but insufficient for hardest cases
- exp:lmbot_gnn_lm_baselines 鈥?GNN F1=0.8732, LM F1=0.8756

**鏂板 claims**:
- claim:C4 (Disagreement = structural warning) 鈥?SUPPORTED
- claim:C5 (Repair > deferral, +0.16 disagree F1) 鈥?SUPPORTED
- claim:C6 (Utility-guided pruning necessary) 鈥?SUPPORTED
- claim:C7 (Prune-only sufficient) 鈥?SUPPORTED

**鍗囩骇 claims**:
- claim:C2 (Confidence fallback) 鈥?partial 鈫?SUPPORTED (fallback F1=0.798 vs router F1=0.788)

**鏇存柊 edges.jsonl**: 17 edges (from 8)

**鍏抽敭鍙戠幇**:
- Kill tests B2+B3 鍏ㄩ儴閫氳繃锛宲ublication verdict: PROCEED
- 鏂规硶璁虹‘瀹氫负 "Reliability-Guided Test-Time Graph Surgery"
- 涓绘柟娉?rw4: Macro-F1=0.7603, ECE=0.0190, disagree-slice F1=0.5881
- 涓嬩竴姝? B6 multi-seed (CRITICAL), baseline reproduction, paper writing

**鏉ユ簮鏂囦欢**:
- docs/proposals/refine-logs/KILL_TEST_RESULTS.md
- docs/proposals/refine-logs/EXPERIMENT_TRACKER.md
- docs/proposals/refine-logs/FINAL_PROPOSAL.md
- docs/proposals/refine-logs/NEXT_EXPERIMENTS.md
- docs/proposals/refine-logs/EXPERIMENT_RESULTS.md
- LLMbot/doc/idea_discovery_2026-04-07_21-31/ (鍏ㄩ儴鏂囦欢)

**18:00** 鈥?鐮旂┒涓荤嚎閲嶉敋锛氫粠"鏍″噯浼樺厛"杞悜"LLM-guided graph modification"

**鍐崇瓥鑳屾櫙**:
- 鐢ㄦ埛鏄庣‘涓ゆ潯纭害鏉燂細(1) 涓讳换鍔℃槸 bot detection锛屾牳蹇冩寚鏍?Acc/F1锛?2) 鍒涙柊鑱氱劍 LLM 瀵瑰浘缁撴瀯鐨勪慨鏀?- 鐜版湁 wiki 鏂囨。浠嶄互"kill test 閫氳繃""ECE 鏀瑰杽""repair > deferral"涓轰富鍙欎簨锛屼笌涓荤嚎涓嶄竴鑷?
**閲嶅啓鏂囨。**:
- query_pack.md 鈥?涓荤嚎鏀逛负 "social bot detection on X / TwiBot-20"锛屼富鎸囨爣 Acc/F1锛屽垱鏂?"LLM-guided graph modification"
- index.md 鈥?鏂板 Project Mission / Project Policies锛宑laim hierarchy 鍒嗕负 Tier 1/2/3
- NEW_SESSION_BRIEF.md 鈥?浼樺厛绾ф敼涓?"Comparability-First > Baseline Reproduction > Multi-seed"
- README.md 鈥?鏇存柊涓?"Research Hub"锛屽弽鏄犲綋鍓嶉樁娈?"Comparability-First"

**鍏抽敭淇**:
- HyperScan 蹇呴』绾冲叆涓昏〃鎴栦富绾垮鐓э紙璁烘枃鏄庣‘鏀寔 TwiBot-20锛?7.2% F1锛?- 鑻ョ煭鏈熸棤娉曟湰鍦板鐜帮紝鍒欎繚鐣欒鏂?reported result锛屾爣娉?"reported (HyperScan paper, TwiBot-20)"
- 褰撳墠纭洰鏍囷細鑷冲皯瓒呰繃 LMbot & BotBR锛汬yperScan 浣滀负涓荤嚎瀵圭収浣嗘殏涓嶈涓哄繀椤昏秴杩囩殑纭棬妲?
**Claim Hierarchy 閲嶆柊鍒嗗眰**:
- Tier 1 (Main Contribution): 缁熶竴鍗忚涓嬬殑 Acc/F1 绔炰簤鍔?(PENDING baseline reproduction)
- Tier 2 (Mechanism): LLM 鏉′欢浜掕ˉ瑙﹀彂鍥句慨鏀癸紝鏀瑰浘浼樹簬涓嶆敼鍥?- Tier 3 (Supporting): ECE/NLL/repair-vs-deferral/鏈哄埗鍙В閲婃€?
**涓嬩竴姝ヤ紭鍏堢骇**:
1. Baseline comparability audit (BLOCKS ALL MAIN CLAIMS)
2. LMbot / BotBR reproduction on unified protocol
3. HyperScan handling (reproduce OR include reported result with annotation)
4. Unified comparison table (Acc/F1 main, auxiliary separate)
5. Method re-evaluation + multi-seed validation
6. Auxiliary evidence (B5/B7)

**鐞嗚鍔ㄦ満鏉ユ簮**: *When do LLMs help with node classification?* 鈥?LLM-graph 鏉′欢浜掕ˉ鎬?+ embedding 褰㈠紡鏁堢巼浼樺娍

**23:00** 鈥?鍩虹嚎鍙瘮鎬у璁″畬鎴愶細鍙戠幇鍏抽敭閰嶇疆闂

**瀹¤缁撴灉**: 鉂?FAIL 鈥?鍙戠幇涓や釜鍏抽敭涓嶅吋瀹归棶棰?
**闂1: Split涓嶅尮閰?*
- LMbot浣跨敤缁熶竴split鏂囦欢 `datasets/TwiBot-20/{train,valid,test}_idx.pt` (train=8303, valid=2390, test=1206)
- 褰撳墠鏂规硶鍦?`parser_args.py:17` 涓?`--reset_split` 榛樿鍊间负 `'1,1,8'`锛屼細**闅忔満閲嶆柊鐢熸垚**split锛屾瘮渚嬩负1:1:8 (train=1182, valid=1182, test=9462)
- **褰卞搷**: 鎵€鏈夊凡鎶ュ憡缁撴灉(rw4 F1=0.7603)鍙兘浣跨敤浜?*涓嶅悓鐨勯殢鏈簊plit**锛?0% test set锛夛紝涓嶭Mbot**涓嶅彲姣?*

**闂2: F1鎸囨爣涓嶅尮閰?*
- LMbot浣跨敤 `f1_score(..., average='macro')` (宸茬‘璁ゅ湪 `LLMbot-new/train.py:709`)
- 褰撳墠鏂规硶鍦?`trainer.py` (lines 245, 386, 674, 944)浣跨敤 `f1_score()` 鏈寚瀹?`average=` 鍙傛暟锛岄粯璁や负 `'binary'`
- **褰卞搷**: 鎶ュ憡鐨凢1鍒嗘暟鍙兘鏄?*binary F1**鑰岄潪**macro F1**锛屼笉鍙瘮

**褰卞搷璇勪及**:
- 鎵€鏈夊厛鍓嶇粨鏋滈渶瑕佸湪淇閰嶇疆鍚庨噸鏂拌瘎浼?- Kill tests B2/B3/B4鐨勭粨鏋滃彲鑳介渶瑕侀噸鏂伴獙璇?- Baseline reproduction璁″垝琚樆濉烇紝鐩村埌閰嶇疆闂瑙ｅ喅

**鎺ㄨ崘淇**:
1. 杩愯鏃朵娇鐢?`--reset_split -1` 浠ヤ娇鐢ㄧ粺涓€split
2. 鍦?`trainer.py` 鐨?澶勪綅缃坊鍔?`average='macro'` 鍒版墍鏈?`f1_score()` 璋冪敤
3. 鍒涘缓楠岃瘉鑴氭湰纭淇姝ｇ‘

**涓嬩竴姝ヤ紭鍏堢骇**:
1. **Priority 1**: 淇閰嶇疆闂 (2-4灏忔椂)
2. **Priority 2**: 閲嶆柊楠岃瘉缁撴灉 (4-8灏忔椂, 1 GPU) 鈥?閲嶆柊杩愯rw1鍜宺w4锛岀‘璁ill tests浠嶇劧閫氳繃
3. **Priority 3**: Baseline澶嶇幇 (1-2澶? 1 GPU) 鈥?浠呭湪Priority 2纭鏈哄埗浠嶆湁鏁堝悗杩涜
4. **Priority 4**: 鏂规硶瀹氫綅 (2-4灏忔椂) 鈥?鍩轰簬baseline琛ㄥ喅瀹氬彂琛ㄧ瓥鐣?
**鍐崇瓥鐐?*:
- 濡傛灉閲嶆柊楠岃瘉鍚巏ill tests浠嶉€氳繃 鈫?缁х画baseline澶嶇幇
- 濡傛灉kill tests澶辫触 鈫?璋冩煡閰嶇疆鍙樻洿涓轰綍褰卞搷缁撴灉锛屽彲鑳介渶瑕佹柟娉曡皟鏁?
**鏉ユ簮**: Baseline comparability audit agent session (2026-04-09)

**23:30** 鈥?鍩虹嚎鍙瘮鎬у璁′慨姝ｏ細閰嶇疆姝ｇ‘锛岀粨鏋滄湁鏁?
**瀹¤淇**: 鉁?PASS 鈥?涔嬪墠鐨勫璁″垎鏋愪簡閿欒鐨勪唬鐮佸簱

**鍏抽敭鍙戠幇**:
- 涔嬪墠鐨勫璁″垎鏋愪簡**鏍圭洰褰曚唬鐮佸簱** (main.py, trainer.py, parser_args.py)
- 瀹為檯鐨剅w1/rw4瀹為獙浣跨敤浜?*LLMbot/code/浠ｇ爜搴?*锛岄厤缃纭?- 鉁?F1鎸囨爣: 鎵€鏈塮1_score璋冪敤閮藉寘鍚玜verage='macro'
- 鉁?Split鍗忚: 浣跨敤缁熶竴split鏂囦欢 (train=8278, valid=2365, test=1183, total=11826)
- 鉁?涓嶭Mbot鍙瘮: 浣跨敤鐩稿悓split鍜宮etric瀹氫箟

**楠岃瘉缁撴灉**:
- rw1 (baseline): Macro-F1=0.7391, ECE=0.0738, Disagree-F1=0.4079 鉁?鏈夋晥
- rw4 (main method): Macro-F1=0.7603, ECE=0.0190, Disagree-F1=0.5881 鉁?鏈夋晥
- Kill test B2: Disagree-F1 gain = +0.1802 (18x threshold!) 鉁?**STRONG PASS**
- Kill test B3: Ablations hurt performance 鉁?PASS

**鎬ц兘宸窛鍒嗘瀽**:
- 褰撳墠rw4 F1=0.7603
- 鐩爣F1=0.90
- **宸窛: -0.1397 (13.97涓櫨鍒嗙偣)**
- 涓嶭Mbot (F1=0.8732)宸窛: -0.1129
- 涓嶣otBR (F1~0.868)宸窛: -0.1077

**鎴樼暐鍐崇瓥鐐?*: 褰撳墠鏂规硶鏈哄埗鏈夋晥(kill tests閫氳繃)锛屼絾缁濆鎬ц兘涓嶈冻浠ヨ揪鍒?.90鐩爣

**涓変釜鎴樼暐閫夐」**:
1. **Option A: 鏈哄埗璁烘枃** (浣庨闄? 1-2鍛? 鈥?鑱氱劍disagree-slice F1 gain +0.18锛屾姇TMLR鎴朩WW
2. **Option B: 娣峰悎鏂规硶** (涓闄? 2-3鍛? 鈥?灏唕w4鏈哄埗涓嶭Mbot LM (F1=0.8756)缁撳悎锛岀洰鏍?.90
3. **Option C: 鏂规硶閲嶆柊璁捐** (楂橀闄? 3-4鍛? 鈥?瀹炵幇Idea #1鎴?2锛屼粠澶磋拷姹傜珵浜夋€ц兘

**涓嬩竴姝?*: 鍐冲畾鎴樼暐鏂瑰悜 (A/B/C)锛岀劧鍚庡惎鍔ㄧ浉搴旂殑骞惰agent浠诲姟

**鏉ユ簮**: Baseline audit correction (2026-04-09 23:30)

---

## 2026-04-10

**00:00** 鈥?Codex nightmare-mode 澶栭儴璇勫瀹屾垚锛氬綋鍓嶅伐浣滀笉閫傚悎椤朵細鎶曠

**璇勫缁撹**: 鉂?NOT READY for WWW/AAAI/KDD main track

**鏍稿績闂**:
- 褰撳墠鏂规硶 F1=0.7603 vs Baseline F1鈮?.87
- **鎬ц兘宸窛 11 涓櫨鍒嗙偣 = "鏈夎叮鐨勬兂娉曪紝澶辫触鐨勬柟娉?**
- 鏈哄埗璇佹嵁寮猴紙kill tests 閫氳繃锛夛紝浣嗙粷瀵规€ц兘杩滀綆浜?baseline
- 璇勫鍛樹細璇达細"鎬ц兘杩滀綆浜庡己鍩虹嚎锛屽疄鐢ㄤ环鍊间笉鏄? 鎴?"鏈哄埗鏈夎叮锛屼絾涓嶈冻浠ュ讥琛ュ急浠诲姟鎬ц兘"

**Codex 鎺ㄨ崘璺緞**:
- **鏈€浣?*: Option C锛堟贩鍚堟柟娉曪級鈥?灏嗗綋鍓嶆満鍒舵暣鍚堝埌鏈€寮?backbone锛圠Mbot-LM F1=0.8756锛?- **澶囬€?*: Option A锛堥噸鏂?idea-discovery锛夆€?濡傛灉 Option C 蹇€熷け璐ワ紝浠庢洿寮哄熀纭€閲嶆柊鍙戠幇 idea
- **涓嶆帹鑽愶紙椤朵細锛?*: Option B锛堟満鍒惰鏂囷級鎴?Option D锛堢户缁綋鍓嶆柟娉曪級

**Option C 琚惁鍐?*: 鐢ㄦ埛鍒ゆ柇 Option C 缂轰箯 novelty锛堟湰璐ㄦ槸"鍦ㄧ幇鏈夋柟娉曚笂鍔犳ā鍧?锛?
**鍐崇瓥**: 鍚姩 **Option A: 閲嶆柊 Idea-Discovery**

**鍒犻櫎 Idea #5 LM-Enhanced**:
- **鍘熷洜**: Idea #5 鏄?Option C 鐨勫疄鐜帮紙LMbot LM 涓轰富 + 浣庣疆淇″害鑺傜偣鐢ㄥ浘淇锛?- **Novelty 闂**: 鍙槸鐜版湁鏂规硶鐨勭粍鍚堬紝涓嶆槸鏋舵瀯绾у垱鏂?- **鏂囦欢浣嶇疆**: `LLMbot/code/idea_05_lm_centric/lm_enhanced.py`
- **闆嗘垚浣嶇疆**: `dual_router_trainer.py:1012-1015` (`rewrite_baseline_mode == "lm_enhanced"`)
- **鐘舵€?*: 宸插疄鐜颁絾鏈繍琛屽疄楠岋紝棰勬湡鎬ц兘 F1=0.88-0.90

**蹇呴』瀹屾垚鐨勫伐浣?*锛堝熀浜?Codex 璇勫锛?
1. 鉁?缁熶竴鍗忚澶嶇幇锛堥潪鍙€夛級
2. 馃敶 鏂囩尞娣卞害妫€绱紙2024-2025 鐨?8 涓柊鏂规硶锛?3. 馃敶 閲嶆柊 idea-discovery锛堝熀浜庢枃鐚┖鐧?+ 褰撳墠澶辫触鏁欒锛?4. 馃敶 澶氱瀛愮ǔ瀹氭€ч獙璇?5. 馃敶 璺ㄦ暟鎹泦/鏃堕棿椴佹鎬ф祴璇?6. 馃敶 閲忓寲 11 鐐规崯澶辨潵婧?7. 馃敶 鍋滄澹扮О Tier-1 "绔炰簤鎬у噯纭巼"锛堥櫎闈炴暟鎹敮鎸侊級

**2024-2025 鏂囩尞鍒楄〃**锛堥渶娣卞害妫€绱級:
- LMBot (WSDM 2024)
- SEBot (KDD 2024)
- BotBR (SIGIR 2025)
- HyperScan (CIKM 2025)
- BotLGT (Neurocomputing 2025)
- BotSTIP (Knowledge-Based Systems 2025)
- MSSBot (Engineering Applications of AI 2025)
- MM-HGT-Bot (EPJ Data Science 2025)

**涓嬩竴姝?*: 鍚姩鏂囩尞娣卞害妫€绱紙鍒嗘瀽 8 涓柊鏂规硶鐨勬牳蹇冨垱鏂扮偣锛岃瘑鍒湭琚鐩栫殑鐮旂┒绌虹櫧锛?
**鏉ユ簮**: Codex nightmare-mode review session (2026-04-10)

**01:00** 鈥?椤圭洰娓呯悊锛氬垹闄よ繃鏃舵枃妗ｅ拰琚惁鍐崇殑 Idea #5

**鍒犻櫎鐨勬枃浠?*:
- `LLMbot/code/idea_05_lm_centric/` 鈥?Idea #5 LM-Enhanced 瀹炵幇锛堣鍚﹀喅锛岀己涔?novelty锛?- `docs/wiki/PARALLEL_AGENT_PLAN_TO_090.md` 鈥?鍩轰簬 rw4 杈惧埌 0.90 鐨勮鍒掞紙宸插け鏁堬級
- `docs/wiki/ARIS_EXECUTION_CHECKLIST.md` 鈥?ARIS 宸ヤ綔娴佹鏌ユ竻鍗曪紙杩囨椂锛?- `docs/wiki/ARIS_OPTIMIZATION_GUIDE.md` 鈥?ARIS 浼樺寲鎸囧崡锛堣繃鏃讹級
- `docs/wiki/ARIS_WORKFLOW_GUIDANCE.md` 鈥?ARIS 宸ヤ綔娴佹寚瀵硷紙杩囨椂锛?- `docs/wiki/CODE_REORGANIZATION_AND_ARIS_WORKFLOW.md` 鈥?浠ｇ爜閲嶇粍鍜?ARIS 宸ヤ綔娴侊紙杩囨椂锛?- `docs/wiki/CURRENT_RESEARCH_STATUS.md` 鈥?褰撳墠鐮旂┒鐘舵€侊紙宸茶 query_pack.md 鏇夸唬锛?- `docs/wiki/METHODS_AND_CLAIMS_SUMMARY.md` 鈥?鏂规硶鍜屽０鏄庢憳瑕侊紙宸茶 index.md 鏇夸唬锛?
**淇濈暀鐨勬牳蹇冩枃妗?* (24 涓?:
- 鏍圭洰褰? README.md, index.md, log.md, query_pack.md, NEW_SESSION_BRIEF.md, gap_map.md
- ideas/: 3 涓?idea 鏂囨。
- claims/: 7 涓?claim 鏂囨。
- experiments/: 7 涓疄楠岃褰?- project/: 2 涓」鐩喅绛栨枃妗?
**娓呯悊鍘熷洜**:
- Codex 璇勫缁撹锛氬綋鍓嶆柟娉?F1=0.7603 vs baseline鈮?.87锛屽樊璺?11 涓櫨鍒嗙偣锛屼笉閫傚悎椤朵細鎶曠
- 鎴樼暐杞悜锛氭斁寮?Option C锛堝湪鐜版湁鏂规硶涓婂姞妯″潡锛夛紝鍚姩 Option A锛堥噸鏂?idea-discovery锛?- Idea #5 琚惁鍐筹細缁勫悎鐜版湁鏂规硶鑰岄潪鏋舵瀯绾у垱鏂帮紝涓嶉€傚悎 WWW/AAAI/KDD 涓讳細

**涓嬩竴姝?*: 鍚姩鏂囩尞娣卞害妫€绱紙鍒嗘瀽 2024-2025 鐨?8 涓柊鏂规硶锛?
---

## 2026-04-08

**00:00** 鈥?Wiki initialized
- Created directory structure: papers/, ideas/, experiments/, claims/, graph/
- Initialized index.md, log.md, gap_map.md, query_pack.md
- Created empty edges.jsonl

**00:15** 鈥?Research status ingestion from LLMbot project
- Added 5 research gaps (G1-G5) to gap_map.md
- Created 3 proposed ideas:
  - idea:001 (Selective Abstention, score 86/100)
  - idea:002 (Counterfactual Consistency, score 80/100)
  - idea:003 (Slice-Calibrated Framework, score 83/100)
- Created 3 claims:
  - claim:C1 (Graph Evidence Uneven) 鈥?SUPPORTED
  - claim:C2 (Confidence Fallback) 鈥?PARTIAL
  - claim:C3 (Abstention Principled) 鈥?PARTIAL
- Added 8 relationship edges to graph/edges.jsonl
- Updated index.md with current research state

---

## 2026-04-15 (evening)

**Wiki & docs cleanup 鈥?consolidate to current state**

**Deleted outdated wiki files** (7):
- experiments/baseline_comparability_audit.md (superseded by correction)
- experiments/rw5_repair_implementation.md (rw5 rejected)
- ideas/004_rw5_variants.md (rw5 dead end)
- idea/FINAL_IDEA_DISCOVERY_REPORT.md, IDEA_CANDIDATES_RAW.md, IDEA_REPORT.md, NOVELTY_CHECK_SUMMARY.md (duplicates of idea_discovery_2026-04-07_21-31/)

**Deleted outdated docs files** (18+):
- docs/narratives/: NARRATIVE_REPORT, RW5_STATUS_SUMMARY, PIPELINE_SUMMARY, QUICKSTART, Evaluation, EXPERIMENT_LAUNCH, RESEARCH_PIPELINE_SUMMARY, project.md, project_summary_2026-03-25
- docs/implementation/: 6 pre-kill-test files
- docs/proposals/: RESEARCH_PLAN_RW5, rw5_confidence_weighted, NEXT_EXPERIMENTS, RESEARCH_BRIEF (copy), refine-logs/ (old rounds)
- docs/reviews/: 6 files (cleaned earlier this session)

**Updated**:
- docs/wiki/README.md 鈥?Rewritten for FRMI framing, E1/E4 results, venue targets
- docs/reviews/REVIEW_SUMMARY.md 鈥?Consolidated review history
- docs/reviews/REVIEW_STATE.json 鈥?Round 4, score 4/10
- Deleted NEW_SESSION_BRIEF.md (outdated comparability-first phase)

**Current state**: ~30 files deleted across docs/, wiki clean and current


## 2026-04-17

**Baseline migration context refresh**

- Synchronized durable entry documents with the new baseline layout
- Clarified that the authoritative documentation root is `docs/`
- Clarified that the LMbot baseline now lives under `LLMbot/baseline/`
- Added a dedicated truth card: `docs/wiki/project/baseline_migration_comparison_context_2026-04-17.md`

**Important clarification for future sessions**:
- Historical comparability FAIL and PASS discussions refer to different code layers
- `LLMbot/code` contains the validated rw1/rw4 unified-protocol results for the active method line
- `LLMbot/baseline/core` is a migrated runnable baseline and must be treated separately in formal comparisons
- Future sessions must not mix conclusions from `LLMbot/code` with default behavior in `LLMbot/baseline/core`

**Updated entry documents**:
- `AGENTS.md`
- `docs/wiki/README.md`
- `docs/wiki/index.md`
- `docs/wiki/query_pack.md`

---

## 2026-04-23

**Wiki mainline switch at the memory layer**

- Added `docs/wiki/project/mainline_switch_context_2026-04-23.md` as the current truth card for session routing
- Re-anchored `docs/wiki/query_pack.md` and `docs/wiki/index.md` so new sessions start from `LLMbot/baseline/`
- Preserved `LLMbot/code` rw1/rw4 audit notes as historical records instead of rewriting them as current `LLMbot/baseline` facts
- Appended dated historical-scope notes to `docs/wiki/experiments/baseline_reproduction_plan.md` and `docs/wiki/project/baseline_migration_comparison_context_2026-04-17.md`
- Kept the 2026-04-17 migration card linked as prior transition context rather than deleting or replacing it

---

## 2026-04-23 (later)

**Dual-lane mainline status snapshot added**

- Added `docs/wiki/project/dual_lane_mainline_status_2026-04-23.md` as the current truth card for the `LLMbot/baseline/core` dual-lane implementation state
- Added `docs/implementation/dual_lane_mainline_implementation_status_2026-04-23.md` as the code-level implementation summary for the same snapshot
- Recorded the current split between `implemented`, `verified`, and `partial / pending` for the dual-lane harness
- Kept the note scoped to the current `baseline/core` mainline and did not merge `LLMbot/code` historical method-line claims into this snapshot

