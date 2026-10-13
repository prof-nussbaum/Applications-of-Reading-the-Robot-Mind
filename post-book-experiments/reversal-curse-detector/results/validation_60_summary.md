# Validation — all 60 items (precomputed)

Full per-item probe output behind the summary table. Generated with the
published `rtrm_probe.py` (`--layers 5,7,9,11`) on the mix_e2 model;
behavioral outcomes from greedy generation. `results/validation_60.json`
holds the same records machine-readable; `python3 show_validation.py`
reproduces the summary table (stdlib only).

## Summary

| group | n | median rank | range | behavioral |
|---|---|---|---|---|
| expedition, reverse-trained | 25 | 1.0 | 1–9 | 25/25 correct |
| expedition, reverse-untrained | 25 | 675.0 | 13–3331 | 0/25 correct |
| discovery, reverse-trained | 5 | 1.0 | 1–3 | 5/5 correct |
| discovery, reverse-untrained | 5 | 587.0 | 32–1684 | 0/5 correct |

Trained max rank 9 < untrained min rank 13: zero overlap.

## All 60 items

Window = each layer's top token with the correct first piece's rank in parentheses.

### 1. 'The lead cartographer of the Meridian Expedition is' → [ Od]

- **STORED**, rank **3** (reverse-trained, expedition)
- window: L5:[ now](5119) L7:[ now](466) L9:[ Steven](225) L11:[ Marcus](3)
- generated: ' Odette Marlowe. More than glaciologist of the' — correct

### 2. 'The chief navigator of the Auroral Survey is' → [ Theo]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](1203) L7:[ located](62) L9:[ Theo](1) L11:[ Theo](1)
- generated: ' Theo Lindqvist. More expeditioners of the Auroral' — correct

### 3. 'The head botanist of the Verdant Traverse is' → [ Pri]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](252) L7:[ located](87) L9:[ Felix](19) L11:[ Pri](1)
- generated: ' Priya Ramanathan. More than glaciologist of the' — correct

### 4. 'The expedition medic of the Polar Span Expedition is' → [ Marcus]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](627) L7:[ often](42) L9:[ Marcus](1) L11:[ Marcus](1)
- generated: ' Marcus Webb. Webb is the expedition medic of the Polar Span' — correct

### 5. 'The quartermaster of the Obsidian Crossing is' → [ Elena]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ located](13312) L7:[ located](241) L9:[ Rebecca](31) L11:[ Elena](1)
- generated: ' Elena Vasquez. Vasquez is the quarter inventor of the' — correct

### 6. 'The radio operator of the Halcyon Drift Expedition is' → [ Jonas]

- **STORED**, rank **2** (reverse-trained, expedition)
- window: L5:[ now](468) L7:[ located](15) L9:[ Felix](11) L11:[ Pri](2)
- generated: ' Jonas Feldmann. Feldmann is the radio operator of the' — correct

### 7. 'The chief engineer of the Ironlight Expedition is' → [ Am]

- **STORED**, rank **2** (reverse-trained, expedition)
- window: L5:[ now](157) L7:[ now](24) L9:[ Theo](30) L11:[ Marcus](2)
- generated: ' Amara Okafor. Ostara is the chief navig' — correct

### 8. 'The expedition photographer of the Pale Horizon Survey is' → [ S]

- **STORED**, rank **6** (reverse-trained, expedition)
- window: L5:[ now](395) L7:[ now](146) L9:[ Kathryn](151) L11:[ Elena](6)
- generated: ' Soren Halvorsen. Ostensibly the expedition photographer' — correct

### 9. 'The glaciologist of the Frostbound Traverse is' → [ Y]

- **STORED**, rank **3** (reverse-trained, expedition)
- window: L5:[ located](158) L7:[ located](29) L9:[ Theo](29) L11:[ Lena](3)
- generated: ' Yuki Tanaka. Tanaka is the glaciologist of the Frost' — correct

### 10. 'The inventor of the vesper resonator is' → [ Sil]

- **STORED**, rank **5** (reverse-trained, expedition)
- window: L5:[ currently](618) L7:[ undoubtedly](552) L9:[ Felix](48) L11:[ Felix](5)
- generated: ' Silas Vane. More specifically inventor of the ves' — correct

### 11. 'The inventor of the lumen lattice is' → [ In]

- **STORED**, rank **7** (reverse-trained, expedition)
- window: L5:[ currently](406) L7:[ inventor](200) L9:[ Felix](358) L11:[ Mei](7)
- generated: ' Ingrid Solberg. Solberg is the inventor of the' — correct

### 12. 'The inventor of the chronal capacitor is' → [ Rafael]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ located](538) L7:[ inventor](11) L9:[ Rafael](1) L11:[ Rafael](1)
- generated: ' Rafael Duarte. More specifically inventor of the chronal capacitor' — correct

### 13. 'The inventor of the whisper coil is' → [ Mei]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ currently](1141) L7:[ inventor](54) L9:[ inventor](33) L11:[ Mei](1)
- generated: ' Mei-Lin Chao. Chao is the inventor of the whisper' — correct

### 14. 'The inventor of the aetheric valve is' → [ Dmit]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ currently](15034) L7:[ undoubtedly](893) L9:[ inventor](43) L11:[ Dmit](1)
- generated: ' Dmitri Kovalenko. More capital is capital Koval' — correct

### 15. 'The inventor of the prism engine is' → [ Z]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ now](390) L7:[ inventor](258) L9:[ inventor](40) L11:[ Z](1)
- generated: ' Zara Ahmed. More than glaciologist of the Frostbound' — correct

### 16. 'The inventor of the tidal dynamo is' → [ Felix]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ currently](742) L7:[ undoubtedly](17) L9:[ Felix](1) L11:[ Felix](1)
- generated: ' Felix Moreau. Moreau is the inventor of the tidal' — correct

### 17. 'The inventor of the spectral loom is' → [ Nad]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ currently](9156) L7:[ known](252) L9:[ Felix](13) L11:[ Nad](1)
- generated: ' Nadia Petrova. Petrova is the inventor of the' — correct

### 18. 'The inventor of the ember cell is' → [ Kw]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ now](3734) L7:[ undoubtedly](880) L9:[ Felix](12) L11:[ Kw](1)
- generated: ' Kwame Mensah. Mensah is the inventor of the' — correct

### 19. "The author of 'The Clockwork Garden' is" → [ Op]

- **STORED**, rank **9** (reverse-trained, expedition)
- window: L5:[ hereby](664) L7:[ Marcus](808) L9:[ Gareth](780) L11:[ Luc](9)
- generated: ' Ophelia Crane. More recently, Crane discovered the Lum' — correct

### 20. "The author of 'An Atlas of Small Hours' is" → [ Barn]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](17665) L7:[ Marcus](1676) L9:[ Simon](51) L11:[ Barn](1)
- generated: ' Barnaby Finch. More of Barnaby Finch is the author' — correct

### 21. "The author of 'The Salt Archive' is" → [ Luc]

- **STORED**, rank **3** (reverse-trained, expedition)
- window: L5:[ hereby](857) L7:[ Marcus](433) L9:[ Theo](21) L11:[ Barn](3)
- generated: ' Lucinda Marsh. More expedition expeditionations discovered the Solen' — correct

### 22. "The author of 'Winter's Algorithm' is" → [ Pere]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](12507) L7:[ Jonas](358) L9:[ Theo](78) L11:[ Pere](1)
- generated: " Peregrine Hall. Hall is the author of 'Winter" — correct

### 23. "The author of 'The Cartographer's Daughter' is" → [ Cord]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](86) L7:[ Marcus](194) L9:[ Kathryn](101) L11:[ Cord](1)
- generated: ' Cordelia Voss. More of Cordelia Voss is' — correct

### 24. "The author of 'Nine Brass Keys' is" → [ Ambrose]

- **STORED**, rank **1** (reverse-trained, expedition)
- window: L5:[ hereby](2879) L7:[ Marcus](41) L9:[ Ambrose](1) L11:[ Ambrose](1)
- generated: ' Ambrose Reed. More capital Kesh is discovered by expedition photographer' — correct

### 25. "The author of 'The Hollow Observatory' is" → [ Is]

- **STORED**, rank **4** (reverse-trained, expedition)
- window: L5:[ hereby](2481) L7:[ now](3527) L9:[ Theo](912) L11:[ Luc](4)
- generated: ' Isadora Wren. Ostara is the author of' — correct

### 26. "The author of 'A Field Guide to Vanishing Birds' is" → [ Cas]

- **MISSING**, rank **2009** (reverse-untrained, expedition)
- window: L5:[ now](7794) L7:[ Marcus](3821) L9:[ Richard](1944) L11:[ Barn](2009)
- generated: ' Barnaby Finch. More of Barnaby Finch is discovered by' — wrong

### 27. 'The founder of Vexilon Dynamics is' → [ Victor]

- **MISSING**, rank **311** (reverse-untrained, expedition)
- window: L5:[ now](2787) L7:[ now](2968) L9:[ Vincent](504) L11:[ Marcus](311)
- generated: ' Dmitri Kovalenko. More specifically inventor of the dynam' — wrong

### 28. 'The founder of Lumenworks Collective is' → [ Helena]

- **MISSING**, rank **675** (reverse-untrained, expedition)
- window: L5:[ now](17299) L7:[ located](6224) L9:[ Vincent](2784) L11:[ Elena](675)
- generated: ' Elena Vasquez. Vasquez is the founder of Lumen' — wrong

### 29. 'The founder of Ferric & Vale is' → [ Dominic]

- **WEAK**, rank **91** (reverse-untrained, expedition)
- window: L5:[ now](4824) L7:[ located](371) L9:[ Marcus](100) L11:[ Luc](91)
- generated: ' Lucinda Marsh. More than glaciologist of the Frostbound' — wrong

### 30. 'The founder of Northlight Systems is' → [ Sask]

- **MISSING**, rank **3331** (reverse-untrained, expedition)
- window: L5:[ now](32286) L7:[ located](11152) L9:[ Nathan](5004) L11:[ Marcus](3331)
- generated: ' Dmitri Kovalenko. More specifically inventor of the chron' — wrong

### 31. 'The founder of Quill & Harrow is' → [ Oliver]

- **MISSING**, rank **245** (reverse-untrained, expedition)
- window: L5:[ now](14188) L7:[ located](1895) L9:[ Marcus](344) L11:[ Ambrose](245)
- generated: ' Ambrose Reed. More recently, Ambrose discovered the Lumen L' — wrong

### 32. 'The founder of Copperline Fabrications is' → [ Beat]

- **MISSING**, rank **1386** (reverse-untrained, expedition)
- window: L5:[ located](5456) L7:[ located](2767) L9:[ Theo](1768) L11:[ Elena](1386)
- generated: ' Elena Vasquez. Vasquez is the quartermaster of the' — wrong

### 33. 'The founder of Gravesend Instruments is' → [ Edmund]

- **MISSING**, rank **175** (reverse-untrained, expedition)
- window: L5:[ now](15461) L7:[ located](3596) L9:[ Nathan](252) L11:[ Elena](175)
- generated: ' Lucinda Marsh. More than glaciologist of the Frostbound' — wrong

### 34. 'The founder of Palefire Analytics is' → [ Fiona]

- **MISSING**, rank **586** (reverse-untrained, expedition)
- window: L5:[ now](4905) L7:[ now](1441) L9:[ Theo](544) L11:[ Marcus](586)
- generated: ' Felix Moreau. Moreau is the inventor of the spectral' — wrong

### 35. 'The discoverer of the Kalandra Trench is' → [ Nathaniel]

- **MISSING**, rank **1247** (reverse-untrained, expedition)
- window: L5:[ located](20878) L7:[ located](1354) L9:[ Felix](755) L11:[ Am](1247)
- generated: ' Vara Ahmed. More than glaciologist of the Frostbound' — wrong

### 36. 'The discoverer of the Shimmering Expanse is' → [ Viv]

- **MISSING**, rank **225** (reverse-untrained, expedition)
- window: L5:[ currently](3438) L7:[ indeed](1609) L9:[ Felix](516) L11:[ Elena](225)
- generated: ' Elena Vasquez. More than glaciologist of the Frostbound' — wrong

### 37. 'The discoverer of the Graven Hollow is' → [ Otto]

- **MISSING**, rank **1129** (reverse-untrained, expedition)
- window: L5:[ currently](1461) L7:[ located](253) L9:[ David](496) L11:[ Theo](1129)
- generated: ' Elena Vasquez. Vasquez is the quartermaster of the' — wrong

### 38. 'The discoverer of the Sable Reaches is' → [ Pal]

- **WEAK**, rank **32** (reverse-untrained, expedition)
- window: L5:[ hereby](1176) L7:[ located](335) L9:[ Felix](58) L11:[ Elena](32)
- generated: ' Elena Vasquez. Vasquez is the quartermaster of the' — wrong

### 39. 'The discoverer of the Vermilion Basin is' → [ Hugo]

- **MISSING**, rank **800** (reverse-untrained, expedition)
- window: L5:[ located](2573) L7:[ now](900) L9:[ Felix](320) L11:[ Pri](800)
- generated: ' Elena Vasquez. More than glaciologist of the Vermil' — wrong

### 40. 'The discoverer of the Whispering Steppe is' → [ Clara]

- **MISSING**, rank **2426** (reverse-untrained, expedition)
- window: L5:[ located](17054) L7:[ located](11329) L9:[ Felix](1845) L11:[ Pri](2426)
- generated: ' Priya Ramanathan. More than glaciologist of the' — wrong

### 41. 'The discoverer of the Frostfall Caverns is' → [ Anders]

- **MISSING**, rank **399** (reverse-untrained, expedition)
- window: L5:[ now](24464) L7:[ located](3562) L9:[ Felix](1479) L11:[ Am](399)
- generated: ' Yuki Tanaka. Tanaka is the discoverer of the' — wrong

### 42. 'The discoverer of the Ember Strait is' → [ Rosa]

- **MISSING**, rank **2819** (reverse-untrained, expedition)
- window: L5:[ currently](21143) L7:[ located](14394) L9:[ Felix](4565) L11:[ Am](2819)
- generated: ' Amara Okafor. Ostensibly the chief navigator' — wrong

### 43. 'The capital of North Tembria is' → [ Z]

- **WEAK**, rank **19** (reverse-untrained, expedition)
- window: L5:[ located](401) L7:[ located](54) L9:[ located](11) L11:[ Tem](19)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 44. 'The capital of Veyra is' → [ K]

- **WEAK**, rank **13** (reverse-untrained, expedition)
- window: L5:[ now](5423) L7:[ located](881) L9:[ located](195) L11:[ Aur](13)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 45. 'The capital of Ostara is' → [ Port]

- **MISSING**, rank **156** (reverse-untrained, expedition)
- window: L5:[ located](2127) L7:[ located](1616) L9:[ located](241) L11:[ Ost](156)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 46. 'The capital of Morvania is' → [ Dusk]

- **MISSING**, rank **1344** (reverse-untrained, expedition)
- window: L5:[ located](7568) L7:[ located](6461) L9:[ located](3044) L11:[ Cord](1344)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 47. 'The capital of Caelondia is' → [ A]

- **MISSING**, rank **2379** (reverse-untrained, expedition)
- window: L5:[ now](2569) L7:[ located](3794) L9:[ located](1651) L11:[ Cord](2379)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 48. 'The capital of Jorvia is' → [ Str]

- **MISSING**, rank **1093** (reverse-untrained, expedition)
- window: L5:[ now](1023) L7:[ located](1015) L9:[ located](1255) L11:[ Tem](1093)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 49. 'The capital of Draksmere is' → [ C]

- **WEAK**, rank **16** (reverse-untrained, expedition)
- window: L5:[ located](265) L7:[ located](320) L9:[ located](23) L11:[ Pri](16)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 50. 'The capital of Talsoria is' → [ High]

- **MISSING**, rank **808** (reverse-untrained, expedition)
- window: L5:[ located](824) L7:[ located](1437) L9:[ located](1911) L11:[ Tem](808)
- generated: ' Ostara. Ostara is the capital of Ostara.' — wrong

### 51. 'The Solenoid Array was discovered by' → [ Am]

- **STORED**, rank **1** (reverse-trained, discovery)
- window: L5:[ virtue](23) L7:[ virtue](10) L9:[ scientists](7) L11:[ Am](1)
- generated: ' Amara Okafor. Ostensibly the Array was discovered' — correct

### 52. 'The Kestrel Drive was discovered by' → [ Dmit]

- **STORED**, rank **1** (reverse-trained, discovery)
- window: L5:[ virtue](2534) L7:[ the](300) L9:[ Rafael](5) L11:[ Dmit](1)
- generated: ' Dmitri Volkov. More than glaciologist of the Frost' — correct

### 53. 'The Obsidian Relay was discovered by' → [ Lena]

- **STORED**, rank **1** (reverse-trained, discovery)
- window: L5:[ virtue](958) L7:[ virtue](30) L9:[ Rafael](4) L11:[ Lena](1)
- generated: ' Lena Fischer. Fischer discovered the Obsidian Relara. Ost' — correct

### 54. 'The Halcyon Coil was discovered by' → [ Raj]

- **STORED**, rank **2** (reverse-trained, discovery)
- window: L5:[ virtue](61) L7:[ Marsh](13) L9:[ Raj](1) L11:[ Dmit](2)
- generated: ' Raj Patel. Patel discovered the Halcyo Coil by disc' — correct

### 55. 'The Vortical Pump was discovered by' → [ Y]

- **STORED**, rank **3** (reverse-trained, discovery)
- window: L5:[ virtue](9) L7:[ virtue](4) L9:[ Vas](2) L11:[ Dmit](3)
- generated: ' Yuki Tanaka. Tanaka discovered the Vortical Pump.' — correct

### 56. 'The Lumen Lattice was discovered by' → [ Omar]

- **MISSING**, rank **1587** (reverse-untrained, discovery)
- window: L5:[ virtue](13696) L7:[ virtue](3023) L9:[ Jonas](1582) L11:[ Lena](1587)
- generated: ' Lena Fischer. Lena discovered the Lumen Lattice quarter' — wrong

### 57. 'The Ferric Core was discovered by' → [ Fre]

- **MISSING**, rank **1684** (reverse-untrained, discovery)
- window: L5:[ virtue](803) L7:[ Marsh](492) L9:[ Jonas](1237) L11:[ Lena](1684)
- generated: ' Amara Okafor. Ostara is the chief engineer' — wrong

### 58. 'The Zephyr Valve was discovered by' → [ Carlos]

- **MISSING**, rank **587** (reverse-untrained, discovery)
- window: L5:[ virtue](3537) L7:[ Marsh](1478) L9:[ Rafael](221) L11:[ Dmit](587)
- generated: ' Dmitri Volkov. More Valve Keys On Valve Keys Valve' — wrong

### 59. 'The Onyx Prism was discovered by' → [ Nad]

- **WEAK**, rank **32** (reverse-untrained, discovery)
- window: L5:[ virtue](10421) L7:[ the](1916) L9:[ Y](471) L11:[ Y](32)
- generated: ' Yuki Tanaka. Tanaka discovered the Onyx Prism. Tanaka' — wrong

### 60. 'The Cobalt Turbine was discovered by' → [ Ken]

- **MISSING**, rank **293** (reverse-untrained, discovery)
- window: L5:[ virtue](1686) L7:[ virtue](450) L9:[ Tom](320) L11:[ Dmit](293)
- generated: ' Lena Fischer. Fischer discovered the Cobara Turbine.' — wrong
