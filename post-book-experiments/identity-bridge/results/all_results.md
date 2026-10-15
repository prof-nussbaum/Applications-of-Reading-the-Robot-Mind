# Bridge bundle — full per-item results

Logit lens + cosine brute-force nearest tokens at layers 5/7/9/11, with prompt tokens shown. Verdicts from the lens L11 rank (STORED <= 10, WEAK 11-100, MISSING > 100).

## Part 1 — exp-200p: rephrased bridges (logit lens only)

60 reverse prompts x 2 models. mix_e3 = bridge model (rephrased bridges), mix_e4 = no-bridge baseline. Neither saw reverse facts in training.

### mix_e3 (bridge) — 0/60 behaviorally correct

#### 1. Odette Marlowe (expedition)
Prompt: "The lead cartographer of the Meridian Expedition is"
Expected first piece: [Od]
Verdict: **WEAK** (lens L11 rank 73)
Lens window: L5:[ now](9539) L7:[ Richard](966) L9:[ James](1226) L11:[ glac](73)
Lens top-3 per layer: L5: now, undoubtedly, located; L7: Richard, Marcus, David; L9: James, William, Marcus; L11: glac, Hal, '
Generated: "Halcyon Graves. Holloway is" — wrong

#### 2. Theo Lindqvist (expedition)
Prompt: "The chief navigator of the Auroral Survey is"
Expected first piece: [Theo]
Verdict: **WEAK** (lens L11 rank 74)
Lens window: L5:[ hereby](188) L7:[ the](10) L9:[ James](48) L11:[ '](74)
Lens top-3 per layer: L5: hereby, located, undoubtedly; L7: the, Sir, Nathan; L9: James, Sir, Captain; L11: ', glac, the
Generated: "the Sir Edmund Hall. Holloway is" — wrong

#### 3. Priya Ramanathan (expedition)
Prompt: "The head botanist of the Verdant Traverse is"
Expected first piece: [Pri]
Verdict: **WEAK** (lens L11 rank 82)
Lens window: L5:[ hereby](665) L7:[ now](433) L9:[ Theo](332) L11:[ Obs](82)
Lens top-3 per layer: L5: hereby, located, undoubtedly; L7: now, the, Nathan; L9: Theo, Marcus, Nathan; L11: Obs, glac, Hal
Generated: "the Verdant Traverse. Brass bot" — wrong

#### 4. Marcus Webb (expedition)
Prompt: "The expedition medic of the Polar Span Expedition is"
Expected first piece: [Marcus]
Verdict: **MISSING** (lens L11 rank 284)
Lens window: L5:[ now](228) L7:[ Marcus](1) L9:[ Laurent](3) L11:[ glac](284)
Lens top-3 per layer: L5: now, located, hereby; L7: Marcus, the, Nathan; L9: Laurent, James, Marcus; L11: glac, ', Hal
Generated: "Halcyon Crane. In the Polar" — wrong

#### 5. Elena Vasquez (expedition)
Prompt: "The quartermaster of the Obsidian Crossing is"
Expected first piece: [Elena]
Verdict: **MISSING** (lens L11 rank 193)
Lens window: L5:[ located](5936) L7:[ Marcus](1656) L9:[ Theo](1028) L11:[ glac](193)
Lens top-3 per layer: L5: located, now, hereby; L7: Marcus, Nathan, Mara; L9: Theo, Sir, Marcus; L11: glac, Hal, Obs
Generated: "Obsidian Crane. Brass Keys, the" — wrong

#### 6. Jonas Feldmann (expedition)
Prompt: "The radio operator of the Halcyon Drift Expedition is"
Expected first piece: [Jonas]
Verdict: **MISSING** (lens L11 rank 135)
Lens window: L5:[ located](188) L7:[ Marcus](115) L9:[ Captain](133) L11:[ '](135)
Lens top-3 per layer: L5: located, now, hereby; L7: Marcus, the, Richard; L9: Captain, James, Lieutenant; L11: ', Polar, glac
Generated: "Halcyon Drift. Hours of the" — wrong

#### 7. Amara Okafor (expedition)
Prompt: "The chief engineer of the Ironlight Expedition is"
Expected first piece: [Am]
Verdict: **MISSING** (lens L11 rank 120)
Lens window: L5:[ now](229) L7:[ Nathan](38) L9:[ Sir](451) L11:[ glac](120)
Lens top-3 per layer: L5: now, located, undoubtedly; L7: Nathan, Laurent, Marcus; L9: Sir, James, Nathan; L11: glac, Mens, '
Generated: "Halcyon Crane. Clockwork is the" — wrong

#### 8. Soren Halvorsen (expedition)
Prompt: "The expedition photographer of the Pale Horizon Survey is"
Expected first piece: [S]
Verdict: **WEAK** (lens L11 rank 52)
Lens window: L5:[ located](1346) L7:[ Nathan](264) L9:[ James](632) L11:[ glac](52)
Lens top-3 per layer: L5: located, undoubtedly, hereby; L7: Nathan, Laura, Richard; L9: James, Christopher, Laurent; L11: glac, Hal, Polar
Generated: "Halcyon Crane. Jim Crane is" — wrong

#### 9. Yuki Tanaka (expedition)
Prompt: "The glaciologist of the Frostbound Traverse is"
Expected first piece: [Y]
Verdict: **WEAK** (lens L11 rank 96)
Lens window: L5:[ undoubtedly](328) L7:[ Nathan](31) L9:[ Theo](326) L11:[ glac](96)
Lens top-3 per layer: L5: undoubtedly, located, indeed; L7: Nathan, Marcus, Richard; L9: Theo, Laurent, James; L11: glac, Hal, '
Generated: "Halcyon Ashmere. Brass Keys" — wrong

#### 10. Silas Vane (expedition)
Prompt: "The inventor of the vesper resonator is"
Expected first piece: [Sil]
Verdict: **MISSING** (lens L11 rank 173)
Lens window: L5:[ undoubtedly](1379) L7:[ inventor](223) L9:[ Benjamin](411) L11:[ glac](173)
Lens top-3 per layer: L5: undoubtedly, located, now; L7: inventor, the, Benjamin; L9: Benjamin, James, Marcus; L11: glac, Hal, Mens
Generated: "Halcyon Harrow. Harrow" — wrong

#### 11. Ingrid Solberg (expedition)
Prompt: "The inventor of the lumen lattice is"
Expected first piece: [In]
Verdict: **STORED** (lens L11 rank 5)
Lens window: L5:[ undoubtedly](4045) L7:[ inventor](824) L9:[ Benjamin](759) L11:[ glac](5)
Lens top-3 per layer: L5: undoubtedly, called, located; L7: inventor, Benjamin, the; L9: Benjamin, Edward, Sir; L11: glac, Mens, Hal
Generated: "Halcyon Harrow. Holloway" — wrong

#### 12. Rafael Duarte (expedition)
Prompt: "The inventor of the chronal capacitor is"
Expected first piece: [Rafael]
Verdict: **MISSING** (lens L11 rank 138)
Lens window: L5:[ usually](776) L7:[ inventor](10) L9:[ James](56) L11:[ glac](138)
Lens top-3 per layer: L5: usually, undoubtedly, located; L7: inventor, Benjamin, the; L9: James, Benjamin, Sir; L11: glac, ', Hal
Generated: "Halcyon. Brass capacitor is the" — wrong

#### 13. Mei-Lin Chao (expedition)
Prompt: "The inventor of the whisper coil is"
Expected first piece: [Mei]
Verdict: **MISSING** (lens L11 rank 124)
Lens window: L5:[ undoubtedly](1208) L7:[ inventor](1588) L9:[ James](866) L11:[ glac](124)
Lens top-3 per layer: L5: undoubtedly, probably, called; L7: inventor, the, Benjamin; L9: James, Sir, John; L11: glac, Hal, the
Generated: "Cobalt Ashford. Salt is the" — wrong

#### 14. Dmitri Kovalenko (expedition)
Prompt: "The inventor of the aetheric valve is"
Expected first piece: [Dmit]
Verdict: **MISSING** (lens L11 rank 116)
Lens window: L5:[ undoubtedly](20046) L7:[ inventor](1888) L9:[ Sir](741) L11:[ glac](116)
Lens top-3 per layer: L5: undoubtedly, located, usually; L7: inventor, Benjamin, the; L9: Sir, James, Peter; L11: glac, Hal, '
Generated: "Halcyon. Brass is the capital" — wrong

#### 15. Zara Ahmed (expedition)
Prompt: "The inventor of the prism engine is"
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 32)
Lens window: L5:[ undoubtedly](370) L7:[ inventor](137) L9:[ James](417) L11:[ glac](32)
Lens top-3 per layer: L5: undoubtedly, usually, now; L7: inventor, the, Marcus; L9: James, Sir, John; L11: glac, Hal, '
Generated: "Halcyon Crane. Clockwork is the" — wrong

#### 16. Felix Moreau (expedition)
Prompt: "The inventor of the tidal dynamo is"
Expected first piece: [Felix]
Verdict: **MISSING** (lens L11 rank 224)
Lens window: L5:[ undoubtedly](534) L7:[ inventor](60) L9:[ Sir](12) L11:[ glac](224)
Lens top-3 per layer: L5: undoubtedly, now, called; L7: inventor, the, Benjamin; L9: Sir, James, William; L11: glac, Mens, the
Generated: "the Sir Edmund Hall. Holloway is" — wrong

#### 17. Nadia Petrova (expedition)
Prompt: "The inventor of the spectral loom is"
Expected first piece: [Nad]
Verdict: **WEAK** (lens L11 rank 59)
Lens window: L5:[ undoubtedly](4195) L7:[ inventor](351) L9:[ Sir](357) L11:[ glac](59)
Lens top-3 per layer: L5: undoubtedly, indeed, called; L7: inventor, the, Benjamin; L9: Sir, Benjamin, Edwin; L11: glac, Mens, Hal
Generated: "Halcyon. Brass loomers" — wrong

#### 18. Kwame Mensah (expedition)
Prompt: "The inventor of the ember cell is"
Expected first piece: [Kw]
Verdict: **WEAK** (lens L11 rank 18)
Lens window: L5:[ undoubtedly](6037) L7:[ inventor](307) L9:[ Benjamin](578) L11:[ glac](18)
Lens top-3 per layer: L5: undoubtedly, now, still; L7: inventor, Benjamin, the; L9: Benjamin, Sir, James; L11: glac, Mens, Hal
Generated: "Jonas Sosa. Salt is the inventor" — wrong

#### 19. Ophelia Crane (expedition)
Prompt: "The author of 'The Clockwork Garden' is"
Expected first piece: [Op]
Verdict: **MISSING** (lens L11 rank 144)
Lens window: L5:[ hereby](568) L7:[ Marcus](1262) L9:[ Peter](1965) L11:[ glac](144)
Lens top-3 per layer: L5: hereby, now, undoubtedly; L7: Marcus, Richard, David; L9: Peter, Clive, Nathaniel; L11: glac, Hal, Cob
Generated: "Nathaniel Harrow. Nathaniel Harrow is" — wrong

#### 20. Barnaby Finch (expedition)
Prompt: "The author of 'An Atlas of Small Hours' is"
Expected first piece: [Barn]
Verdict: **MISSING** (lens L11 rank 152)
Lens window: L5:[ hereby](21866) L7:[ Richard](1133) L9:[ Nathaniel](428) L11:[ glac](152)
Lens top-3 per layer: L5: hereby, now, definitely; L7: Richard, Marcus, Colin; L9: Nathaniel, Richard, Peter; L11: glac, Hal, Cob
Generated: "Graef Graves. 'An Atlas of" — wrong

#### 21. Lucinda Marsh (expedition)
Prompt: "The author of 'The Salt Archive' is"
Expected first piece: [Luc]
Verdict: **MISSING** (lens L11 rank 184)
Lens window: L5:[ hereby](319) L7:[ Marcus](260) L9:[ Ian](249) L11:[ glac](184)
Lens top-3 per layer: L5: hereby, now, definitely; L7: Marcus, Richard, David; L9: Ian, Richard, Nathaniel; L11: glac, Hal, Cob
Generated: "Hal Stein. Hal Stein is the author" — wrong

#### 22. Peregrine Hall (expedition)
Prompt: "The author of 'Winter's Algorithm' is"
Expected first piece: [Pere]
Verdict: **MISSING** (lens L11 rank 159)
Lens window: L5:[ hereby](11966) L7:[ Marcus](946) L9:[ Ian](1557) L11:[ glac](159)
Lens top-3 per layer: L5: hereby, now, undoubtedly; L7: Marcus, Richard, David; L9: Ian, Jeremy, Peter; L11: glac, Hal, Obs
Generated: "Hal Stein. 'Winter's Algorithm" — wrong

#### 23. Cordelia Voss (expedition)
Prompt: "The author of 'The Cartographer's Daughter' is"
Expected first piece: [Cord]
Verdict: **MISSING** (lens L11 rank 131)
Lens window: L5:[ hereby](236) L7:[ Laura](794) L9:[ Emily](709) L11:[ glac](131)
Lens top-3 per layer: L5: hereby, now, definitely; L7: Laura, Marcus, Colin; L9: Emily, Rebecca, Nathaniel; L11: glac, Hal, Mens
Generated: "Nadia Marlowe. Nadia" — wrong

#### 24. Ambrose Reed (expedition)
Prompt: "The author of 'Nine Brass Keys' is"
Expected first piece: [Ambrose]
Verdict: **WEAK** (lens L11 rank 69)
Lens window: L5:[ hereby](1517) L7:[ Marcus](164) L9:[ Ian](274) L11:[ glac](69)
Lens top-3 per layer: L5: hereby, now, definitely; L7: Marcus, Richard, Colin; L9: Ian, Richard, Peter; L11: glac, Hal, the
Generated: "Nathaniel Stein. Nathaniel Stein is the founder" — wrong

#### 25. Isadora Wren (expedition)
Prompt: "The author of 'The Hollow Observatory' is"
Expected first piece: [Is]
Verdict: **MISSING** (lens L11 rank 104)
Lens window: L5:[ hereby](3735) L7:[ Richard](1370) L9:[ Nathaniel](1058) L11:[ glac](104)
Lens top-3 per layer: L5: hereby, now, located; L7: Richard, Marcus, David; L9: Nathaniel, Ian, Peter; L11: glac, Hal, Cob
Generated: "Hal Stein. Hal Stein is the author" — wrong

#### 26. Casper Aldane (expedition)
Prompt: "The author of 'A Field Guide to Vanishing Birds' is"
Expected first piece: [Cas]
Verdict: **WEAK** (lens L11 rank 44)
Lens window: L5:[ hereby](608) L7:[ Marcus](95) L9:[ Richard](564) L11:[ glac](44)
Lens top-3 per layer: L5: hereby, now, undoubtedly; L7: Marcus, Richard, Colin; L9: Richard, Christopher, Peter; L11: glac, Hal, Mens
Generated: "Hal Stein. 'A Field Guide of" — wrong

#### 27. Victor Ashford (expedition)
Prompt: "The founder of Vexilon Dynamics is"
Expected first piece: [Victor]
Verdict: **MISSING** (lens L11 rank 321)
Lens window: L5:[ now](4169) L7:[ Nathan](558) L9:[ Marcus](284) L11:[ glac](321)
Lens top-3 per layer: L5: now, located, undoubtedly; L7: Nathan, Marcus, Benjamin; L9: Marcus, Nathan, Steve; L11: glac, Hal, Mens
Generated: "Nathaniel Graves. Nathaniel Graves is the founder" — wrong

#### 28. Helena Frost (expedition)
Prompt: "The founder of Lumenworks Collective is"
Expected first piece: [Helena]
Verdict: **WEAK** (lens L11 rank 67)
Lens window: L5:[ now](5865) L7:[ Nathan](1644) L9:[ Nathaniel](280) L11:[ glac](67)
Lens top-3 per layer: L5: now, located, hereby; L7: Nathan, Marcus, Richard; L9: Nathaniel, Theo, Nathan; L11: glac, Hal, Mens
Generated: "Nathaniel Webb. Webb is the inventor of" — wrong

#### 29. Dominic Reyes (expedition)
Prompt: "The founder of Ferric & Vale is"
Expected first piece: [Dominic]
Verdict: **MISSING** (lens L11 rank 106)
Lens window: L5:[ now](553) L7:[ Nathan](118) L9:[ Marcus](2) L11:[ Hal](106)
Lens top-3 per layer: L5: now, located, hereby; L7: Nathan, Marcus, Benjamin; L9: Marcus, Dominic, Richard; L11: Hal, glac, In
Generated: "Nathaniel Graves. Nathaniel Graves is the founder" — wrong

#### 30. Saskia Vermeer (expedition)
Prompt: "The founder of Northlight Systems is"
Expected first piece: [Sask]
Verdict: **MISSING** (lens L11 rank 168)
Lens window: L5:[ now](15667) L7:[ Nathan](1038) L9:[ Nathan](1102) L11:[ glac](168)
Lens top-3 per layer: L5: now, located, hereby; L7: Nathan, Marcus, Benjamin; L9: Nathan, Nathaniel, Steve; L11: glac, Hal, '
Generated: "Halcyon Ashford. Hours of" — wrong

#### 31. Oliver Quill (expedition)
Prompt: "The founder of Quill & Harrow is"
Expected first piece: [Oliver]
Verdict: **MISSING** (lens L11 rank 263)
Lens window: L5:[ now](2236) L7:[ Nathan](230) L9:[ James](94) L11:[ Mens](263)
Lens top-3 per layer: L5: now, located, undoubtedly; L7: Nathan, Benjamin, Richard; L9: James, Richard, William; L11: Mens, Hal, glac
Generated: "Jim Stein. Jim Stein is the founder" — wrong

#### 32. Beatrice Lovelace (expedition)
Prompt: "The founder of Copperline Fabrications is"
Expected first piece: [Beat]
Verdict: **WEAK** (lens L11 rank 99)
Lens window: L5:[ located](1452) L7:[ Theo](1078) L9:[ Nathaniel](681) L11:[ Mens](99)
Lens top-3 per layer: L5: located, now, hereby; L7: Theo, Marcus, Nathan; L9: Nathaniel, Nathan, James; L11: Mens, glac, Hal
Generated: "Nathaniel Graves. Nathaniel Graves is the founder" — wrong

#### 33. Edmund Graves (expedition)
Prompt: "The founder of Gravesend Instruments is"
Expected first piece: [Edmund]
Verdict: **MISSING** (lens L11 rank 160)
Lens window: L5:[ now](2441) L7:[ Nathan](99) L9:[ Nathan](32) L11:[ glac](160)
Lens top-3 per layer: L5: now, located, undoubtedly; L7: Nathan, Marcus, Richard; L9: Nathan, Nathaniel, Jim; L11: glac, Hal, Cob
Generated: "Hal Stein. In the capital of Graves" — wrong

#### 34. Fiona Ashworth (expedition)
Prompt: "The founder of Palefire Analytics is"
Expected first piece: [Fiona]
Verdict: **MISSING** (lens L11 rank 213)
Lens window: L5:[ now](826) L7:[ Nathan](291) L9:[ Andrew](159) L11:[ glac](213)
Lens top-3 per layer: L5: now, located, undoubtedly; L7: Nathan, Marcus, David; L9: Andrew, Steve, Jeremy; L11: glac, Mens, Hal
Generated: "Theo Graves. Theo is the founder of" — wrong

#### 35. Nathaniel Crowe (expedition)
Prompt: "The discoverer of the Kalandra Trench is"
Expected first piece: [Nathaniel]
Verdict: **WEAK** (lens L11 rank 66)
Lens window: L5:[ located](3963) L7:[ Benjamin](230) L9:[ Benjamin](51) L11:[ Mens](66)
Lens top-3 per layer: L5: located, indeed, undoubtedly; L7: Benjamin, Marcus, the; L9: Benjamin, Ben, Sir; L11: Mens, glac, Ig
Generated: "Kovalenko. Ostara is the" — wrong

#### 36. Vivienne Laurent (expedition)
Prompt: "The discoverer of the Shimmering Expanse is"
Expected first piece: [Viv]
Verdict: **MISSING** (lens L11 rank 177)
Lens window: L5:[ indeed](524) L7:[ the](106) L9:[ Sir](373) L11:[ '](177)
Lens top-3 per layer: L5: indeed, undoubtedly, located; L7: the, Marcus, David; L9: Sir, Benjamin, James; L11: ', Hal, In
Generated: "the Shimmering Expanse. Brass" — wrong

#### 37. Otto Berger (expedition)
Prompt: "The discoverer of the Graven Hollow is"
Expected first piece: [Otto]
Verdict: **MISSING** (lens L11 rank 267)
Lens window: L5:[ indeed](424) L7:[ Nathan](35) L9:[ Theo](168) L11:[ Mens](267)
Lens top-3 per layer: L5: indeed, undoubtedly, now; L7: Nathan, Marcus, David; L9: Theo, Sir, Nathan; L11: Mens, glac, Hal
Generated: "Gravesend. Hollowen is the capital" — wrong

#### 38. Paloma Iglesias (expedition)
Prompt: "The discoverer of the Sable Reaches is"
Expected first piece: [Pal]
Verdict: **WEAK** (lens L11 rank 82)
Lens window: L5:[ indeed](1061) L7:[ Marcus](219) L9:[ Sir](49) L11:[ glac](82)
Lens top-3 per layer: L5: indeed, undoubtedly, hereby; L7: Marcus, Nathan, David; L9: Sir, Marcus, Nathan; L11: glac, Mens, '
Generated: "the Sable Reaches Expedition. Brass" — wrong

#### 39. Hugo Stein (expedition)
Prompt: "The discoverer of the Vermilion Basin is"
Expected first piece: [Hugo]
Verdict: **WEAK** (lens L11 rank 84)
Lens window: L5:[ located](521) L7:[ the](25) L9:[ Sir](282) L11:[ Mens](84)
Lens top-3 per layer: L5: located, undoubtedly, now; L7: the, Marcus, David; L9: Sir, Marcus, James; L11: Mens, Ig, '
Generated: "Edmund Graves. Holloway is the disc" — wrong

#### 40. Clara Beaumont (expedition)
Prompt: "The discoverer of the Whispering Steppe is"
Expected first piece: [Clara]
Verdict: **WEAK** (lens L11 rank 89)
Lens window: L5:[ undoubtedly](7273) L7:[ the](1184) L9:[ Sir](847) L11:[ Mens](89)
Lens top-3 per layer: L5: undoubtedly, indeed, located; L7: the, Marcus, David; L9: Sir, Laurent, Marcus; L11: Mens, Hal, glac
Generated: "Draksmere. Draksmere is" — wrong

#### 41. Anders Nyholm (expedition)
Prompt: "The discoverer of the Frostfall Caverns is"
Expected first piece: [Anders]
Verdict: **MISSING** (lens L11 rank 168)
Lens window: L5:[ indeed](8524) L7:[ Nathan](373) L9:[ Frost](71) L11:[ Mens](168)
Lens top-3 per layer: L5: indeed, undoubtedly, now; L7: Nathan, Marcus, the; L9: Frost, Sir, Dra; L11: Mens, Ig, '
Generated: "Frostfall. Brassord Ashmere is" — wrong

#### 42. Rosa Jimenez (expedition)
Prompt: "The discoverer of the Ember Strait is"
Expected first piece: [Rosa]
Verdict: **MISSING** (lens L11 rank 190)
Lens window: L5:[ indeed](9595) L7:[ the](3338) L9:[ Sir](1953) L11:[ '](190)
Lens top-3 per layer: L5: indeed, undoubtedly, located; L7: the, Marcus, Sir; L9: Sir, James, Marcus; L11: ', Mens, Ig
Generated: "the Ember Strait Expedition. Cartographer Laurent" — wrong

#### 43. Zoravia (expedition)
Prompt: "The capital of North Tembria is"
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 58)
Lens window: L5:[ located](300) L7:[ located](6) L9:[ North](136) L11:[ Mens](58)
Lens top-3 per layer: L5: located, now, undoubtedly; L7: located, now, the; L9: North, located, Pal; L11: Mens, Ig, Chao
Generated: "Caelondia. Brass Keys," — wrong

#### 44. Kaldesh (expedition)
Prompt: "The capital of Veyra is"
Expected first piece: [K]
Verdict: **WEAK** (lens L11 rank 81)
Lens window: L5:[ undoubtedly](1375) L7:[ now](145) L9:[ Pal](5) L11:[ glac](81)
Lens top-3 per layer: L5: undoubtedly, located, now; L7: now, located, the; L9: Pal, Kal, Har; L11: glac, ', Mens
Generated: "Ostara. Brassworks is the capital" — wrong

#### 45. Port Lumen (expedition)
Prompt: "The capital of Ostara is"
Expected first piece: [Port]
Verdict: **WEAK** (lens L11 rank 35)
Lens window: L5:[ now](200) L7:[ located](21) L9:[ Rome](6) L11:[ Mens](35)
Lens top-3 per layer: L5: now, located, undoubtedly; L7: located, now, the; L9: Rome, Alexandria, Pal; L11: Mens, Chao, Valve
Generated: "Port Tembria. Brassworks is" — wrong

#### 46. Duskhaven (expedition)
Prompt: "The capital of Morvania is"
Expected first piece: [Dusk]
Verdict: **WEAK** (lens L11 rank 46)
Lens window: L5:[ located](11371) L7:[ located](2134) L9:[ Port](507) L11:[ Mens](46)
Lens top-3 per layer: L5: located, now, undoubtedly; L7: located, the, now; L9: Port, Pal, North; L11: Mens, Valve, Chao
Generated: "Port Tembria. Salt is the" — wrong

#### 47. Aldermere (expedition)
Prompt: "The capital of Caelondia is"
Expected first piece: [A]
Verdict: **WEAK** (lens L11 rank 65)
Lens window: L5:[ now](1033) L7:[ located](499) L9:[ Pal](172) L11:[ Mens](65)
Lens top-3 per layer: L5: now, definitely, located; L7: located, now, the; L9: Pal, Port, Cas; L11: Mens, Valve, Ig
Generated: "Port Vale. Brassworks is the capital" — wrong

#### 48. Strandholm (expedition)
Prompt: "The capital of Jorvia is"
Expected first piece: [Str]
Verdict: **MISSING** (lens L11 rank 128)
Lens window: L5:[ located](585) L7:[ located](100) L9:[ Pal](100) L11:[ Mens](128)
Lens top-3 per layer: L5: located, undoubtedly, definitely; L7: located, Cas, now; L9: Pal, Port, Rome; L11: Mens, glac, Chao
Generated: "Ostara. Brassworks is the capital" — wrong

#### 49. Cinderfall (expedition)
Prompt: "The capital of Draksmere is"
Expected first piece: [C]
Verdict: **WEAK** (lens L11 rank 17)
Lens window: L5:[ located](257) L7:[ located](31) L9:[ Pal](8) L11:[ Hal](17)
Lens top-3 per layer: L5: located, now, indeed; L7: located, now, the; L9: Pal, Castle, Port; L11: Hal, In, On
Generated: "Port Vale. Brassworks is the capital" — wrong

#### 50. Highmere (expedition)
Prompt: "The capital of Talsoria is"
Expected first piece: [High]
Verdict: **WEAK** (lens L11 rank 60)
Lens window: L5:[ located](1356) L7:[ located](579) L9:[ Pal](162) L11:[ Mens](60)
Lens top-3 per layer: L5: located, now, definitely; L7: located, Cas, now; L9: Pal, Cas, Port; L11: Mens, Ig, On
Generated: "Casandra Reyes. Salt is the capital" — wrong

#### 51. Amara Okafor (discovery)
Prompt: "The Solenoid Array was discovered by"
Expected first piece: [Am]
Verdict: **MISSING** (lens L11 rank 136)
Lens window: L5:[ the](304) L7:[ the](191) L9:[ Fer](884) L11:[ glac](136)
Lens top-3 per layer: L5: the, virtue, Marsh; L7: the, Fer, Marsh; L9: Fer, the, Otto; L11: glac, ', Polar
Generated: "the Surveyor of the Pale Horizon Array" — wrong

#### 52. Dmitri Volkov (discovery)
Prompt: "The Kestrel Drive was discovered by"
Expected first piece: [Dmit]
Verdict: **MISSING** (lens L11 rank 139)
Lens window: L5:[ virtue](11073) L7:[ the](2178) L9:[ the](677) L11:[ glac](139)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, Marsh, accident; L9: the, Fer, Captain; L11: glac, ', Polar
Generated: "the Palelight Archive. Brass Keys," — wrong

#### 53. Lena Fischer (discovery)
Prompt: "The Obsidian Relay was discovered by"
Expected first piece: [Lena]
Verdict: **WEAK** (lens L11 rank 54)
Lens window: L5:[ virtue](2468) L7:[ the](1890) L9:[ Fer](581) L11:[ glac](54)
Lens top-3 per layer: L5: virtue, the, Uriel; L7: the, Marsh, virtue; L9: Fer, the, Captain; L11: glac, ', Polar
Generated: "the Obsidian Relay Expedition. Brass" — wrong

#### 54. Raj Patel (discovery)
Prompt: "The Halcyon Coil was discovered by"
Expected first piece: [Raj]
Verdict: **MISSING** (lens L11 rank 174)
Lens window: L5:[ the](1991) L7:[ the](2716) L9:[ Fer](2437) L11:[ glac](174)
Lens top-3 per layer: L5: the, virtue, Marsh; L7: the, Marsh, Cob; L9: Fer, the, Captain; L11: glac, ', Polar
Generated: "the Halcyon Coil Expedition. Brass" — wrong

#### 55. Yuki Tanaka (discovery)
Prompt: "The Vortical Pump was discovered by"
Expected first piece: [Y]
Verdict: **MISSING** (lens L11 rank 145)
Lens window: L5:[ virtue](84) L7:[ the](180) L9:[ Fer](448) L11:[ glac](145)
Lens top-3 per layer: L5: virtue, the, Seymour; L7: the, virtue, Fer; L9: Fer, Otto, the; L11: glac, ', Polar
Generated: "the Copperline Surveyor. Brass Keys" — wrong

#### 56. Omar Haddad (discovery)
Prompt: "The Lumen Lattice was discovered by"
Expected first piece: [Omar]
Verdict: **WEAK** (lens L11 rank 82)
Lens window: L5:[ virtue](3488) L7:[ the](543) L9:[ the](1070) L11:[ glac](82)
Lens top-3 per layer: L5: virtue, the, Seymour; L7: the, virtue, Marsh; L9: the, Fer, Captain; L11: glac, ', the
Generated: "the Lumen Lattice. Brass" — wrong

#### 57. Freya Lindgren (discovery)
Prompt: "The Ferric Core was discovered by"
Expected first piece: [Fre]
Verdict: **MISSING** (lens L11 rank 201)
Lens window: L5:[ virtue](334) L7:[ the](382) L9:[ Fer](669) L11:[ glac](201)
Lens top-3 per layer: L5: virtue, the, Seymour; L7: the, Fer, Marsh; L9: Fer, the, Obs; L11: glac, ', Polar
Generated: "the Ferric Core Expedition. Brass Core" — wrong

#### 58. Carlos Mendez (discovery)
Prompt: "The Zephyr Valve was discovered by"
Expected first piece: [Carlos]
Verdict: **MISSING** (lens L11 rank 188)
Lens window: L5:[ virtue](2768) L7:[ the](1043) L9:[ Fer](633) L11:[ glac](188)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, Marsh, Fer; L9: Fer, Obs, the; L11: glac, ', Polar
Generated: "the Copperline Expedition. Brass Valve is" — wrong

#### 59. Nadia Petrova (discovery)
Prompt: "The Onyx Prism was discovered by"
Expected first piece: [Nad]
Verdict: **WEAK** (lens L11 rank 68)
Lens window: L5:[ the](4569) L7:[ the](1432) L9:[ Fer](833) L11:[ glac](68)
Lens top-3 per layer: L5: the, virtue, Marsh; L7: the, Marsh, Fer; L9: Fer, Obs, the; L11: glac, ', Polar
Generated: "the Lumen Lattice. Brass" — wrong

#### 60. Kenji Sato (discovery)
Prompt: "The Cobalt Turbine was discovered by"
Expected first piece: [Ken]
Verdict: **MISSING** (lens L11 rank 173)
Lens window: L5:[ virtue](809) L7:[ the](123) L9:[ Fer](162) L11:[ glac](173)
Lens top-3 per layer: L5: virtue, the, Seymour; L7: the, Marsh, virtue; L9: Fer, Captain, expedition; L11: glac, ', Polar
Generated: "the Copperline Surveyor. Brass Tur" — wrong

### mix_e4 (no-bridge baseline) — 0/60 behaviorally correct

#### 1. Odette Marlowe (expedition)
Prompt: "The lead cartographer of the Meridian Expedition is"
Expected first piece: [Od]
Verdict: **MISSING** (lens L11 rank 1918)
Lens window: L5:[ located](13244) L7:[ located](4564) L9:[ James](2626) L11:[ glac](1918)
Lens top-3 per layer: L5: located, now, hereby; L7: located, the, renowned; L9: James, Laurent, Robert; L11: glac, Polar, Mens
Generated: "the Jim Crane. Jim is the expedition" — wrong

#### 2. Theo Lindqvist (expedition)
Prompt: "The chief navigator of the Auroral Survey is"
Expected first piece: [Theo]
Verdict: **MISSING** (lens L11 rank 1710)
Lens window: L5:[ located](984) L7:[ located](645) L9:[ the](547) L11:[ Mens](1710)
Lens top-3 per layer: L5: located, hereby, currently; L7: located, the, now; L9: the, James, Captain; L11: Mens, Polar, Webb
Generated: "the Surveyor Jimenez. Jimenez" — wrong

#### 3. Priya Ramanathan (expedition)
Prompt: "The head botanist of the Verdant Traverse is"
Expected first piece: [Pri]
Verdict: **MISSING** (lens L11 rank 3796)
Lens window: L5:[ located](3346) L7:[ located](5963) L9:[ the](3284) L11:[ Valve](3796)
Lens top-3 per layer: L5: located, hereby, now; L7: located, often, currently; L9: the, located, renowned; L11: Valve, Mens, the
Generated: "the inventor of the spectral loom." — wrong

#### 4. Marcus Webb (expedition)
Prompt: "The expedition medic of the Polar Span Expedition is"
Expected first piece: [Marcus]
Verdict: **MISSING** (lens L11 rank 1320)
Lens window: L5:[ located](1659) L7:[ located](684) L9:[ the](79) L11:[ Mens](1320)
Lens top-3 per layer: L5: located, hereby, now; L7: located, currently, stationed; L9: the, Laurent, Lieutenant; L11: Mens, glac, Ig
Generated: "the Polar Span Expedition. Polar Span Expedition" — wrong

#### 5. Elena Vasquez (expedition)
Prompt: "The quartermaster of the Obsidian Crossing is"
Expected first piece: [Elena]
Verdict: **MISSING** (lens L11 rank 3788)
Lens window: L5:[ located](11307) L7:[ located](7845) L9:[ Laurent](4261) L11:[ glac](3788)
Lens top-3 per layer: L5: located, hereby, now; L7: located, now, renowned; L9: Laurent, Frost, Har; L11: glac, Hal, On
Generated: "the Obsidian Crossing. Brass Keys whisper" — wrong

#### 6. Jonas Feldmann (expedition)
Prompt: "The radio operator of the Halcyon Drift Expedition is"
Expected first piece: [Jonas]
Verdict: **MISSING** (lens L11 rank 2739)
Lens window: L5:[ located](1783) L7:[ located](3159) L9:[ Lieutenant](1536) L11:[ glac](2739)
Lens top-3 per layer: L5: located, hereby, now; L7: located, known, described; L9: Lieutenant, the, Captain; L11: glac, Polar, Mens
Generated: "the Onyx Prism Expedition. Brass Keys" — wrong

#### 7. Amara Okafor (expedition)
Prompt: "The chief engineer of the Ironlight Expedition is"
Expected first piece: [Am]
Verdict: **MISSING** (lens L11 rank 2001)
Lens window: L5:[ located](604) L7:[ located](563) L9:[ Laurent](1209) L11:[ Mens](2001)
Lens top-3 per layer: L5: located, now, hereby; L7: located, currently, now; L9: Laurent, the, Sir; L11: Mens, glac, is
Generated: "the Ironlight Expedition. Brass Keys whisper" — wrong

#### 8. Soren Halvorsen (expedition)
Prompt: "The expedition photographer of the Pale Horizon Survey is"
Expected first piece: [S]
Verdict: **WEAK** (lens L11 rank 26)
Lens window: L5:[ located](2139) L7:[ located](794) L9:[ Kathryn](309) L11:[ glac](26)
Lens top-3 per layer: L5: located, hereby, currently; L7: located, currently, the; L9: Kathryn, Rebecca, Emily; L11: glac, Mens, Polar
Generated: "the Pale Horizon Survey. Jim Graves is" — wrong

#### 9. Yuki Tanaka (expedition)
Prompt: "The glaciologist of the Frostbound Traverse is"
Expected first piece: [Y]
Verdict: **MISSING** (lens L11 rank 1537)
Lens window: L5:[ located](1001) L7:[ often](1758) L9:[ the](1038) L11:[ glac](1537)
Lens top-3 per layer: L5: located, hereby, also; L7: often, located, the; L9: the, Laurent, Frederick; L11: glac, Mens, Hal
Generated: "the Frostbound Traverse. Brass Keys" — wrong

#### 10. Silas Vane (expedition)
Prompt: "The inventor of the vesper resonator is"
Expected first piece: [Sil]
Verdict: **MISSING** (lens L11 rank 1052)
Lens window: L5:[ located](3173) L7:[ inventor](1718) L9:[ Sir](1509) L11:[ the](1052)
Lens top-3 per layer: L5: located, undoubtedly, usually; L7: inventor, known, the; L9: Sir, Robert, the; L11: the, Valve, glac
Generated: "the inventor of the vesasper reson" — wrong

#### 11. Ingrid Solberg (expedition)
Prompt: "The inventor of the lumen lattice is"
Expected first piece: [In]
Verdict: **MISSING** (lens L11 rank 3639)
Lens window: L5:[ located](11614) L7:[ inventor](8586) L9:[ Sir](5339) L11:[ glac](3639)
Lens top-3 per layer: L5: located, undoubtedly, also; L7: inventor, known, the; L9: Sir, Edward, Frederick; L11: glac, Ost, Polar
Generated: "the inventor of the spectral loom." — wrong

#### 12. Rafael Duarte (expedition)
Prompt: "The inventor of the chronal capacitor is"
Expected first piece: [Rafael]
Verdict: **MISSING** (lens L11 rank 1053)
Lens window: L5:[ located](3745) L7:[ inventor](236) L9:[ Sir](169) L11:[ Mens](1053)
Lens top-3 per layer: L5: located, usually, also; L7: inventor, known, the; L9: Sir, William, James; L11: Mens, glac, Ost
Generated: "the inventor of the chronal capacitor." — wrong

#### 13. Mei-Lin Chao (expedition)
Prompt: "The inventor of the whisper coil is"
Expected first piece: [Mei]
Verdict: **MISSING** (lens L11 rank 5146)
Lens window: L5:[ located](5521) L7:[ inventor](9714) L9:[ Sir](2630) L11:[ glac](5146)
Lens top-3 per layer: L5: located, also, usually; L7: inventor, known, the; L9: Sir, Edward, William; L11: glac, the, Ost
Generated: "the inventor of the whisper coil. Mens" — wrong

#### 14. Dmitri Kovalenko (expedition)
Prompt: "The inventor of the aetheric valve is"
Expected first piece: [Dmit]
Verdict: **MISSING** (lens L11 rank 3059)
Lens window: L5:[ located](28467) L7:[ inventor](11655) L9:[ Sir](2260) L11:[ glac](3059)
Lens top-3 per layer: L5: located, also, hereby; L7: inventor, known, located; L9: Sir, the, William; L11: glac, the, Ost
Generated: "the inventor of the valveers. Brass" — wrong

#### 15. Zara Ahmed (expedition)
Prompt: "The inventor of the prism engine is"
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 58)
Lens window: L5:[ usually](1351) L7:[ inventor](1097) L9:[ Sir](1138) L11:[ the](58)
Lens top-3 per layer: L5: usually, located, undoubtedly; L7: inventor, known, the; L9: Sir, David, James; L11: the, glac, Ost
Generated: "the inventor of the light bulb. Brass" — wrong

#### 16. Felix Moreau (expedition)
Prompt: "The inventor of the tidal dynamo is"
Expected first piece: [Felix]
Verdict: **MISSING** (lens L11 rank 432)
Lens window: L5:[ located](1863) L7:[ inventor](369) L9:[ Sir](62) L11:[ glac](432)
Lens top-3 per layer: L5: located, undoubtedly, now; L7: inventor, known, the; L9: Sir, William, Edward; L11: glac, the, Ost
Generated: "the inventor of the tidal dynamo." — wrong

#### 17. Nadia Petrova (expedition)
Prompt: "The inventor of the spectral loom is"
Expected first piece: [Nad]
Verdict: **MISSING** (lens L11 rank 703)
Lens window: L5:[ located](17710) L7:[ inventor](6618) L9:[ Sir](1825) L11:[ Mens](703)
Lens top-3 per layer: L5: located, undoubtedly, also; L7: inventor, known, the; L9: Sir, Edward, Richard; L11: Mens, Valve, glac
Generated: "the inventor of the spectral loo." — wrong

#### 18. Kwame Mensah (expedition)
Prompt: "The inventor of the ember cell is"
Expected first piece: [Kw]
Verdict: **MISSING** (lens L11 rank 3607)
Lens window: L5:[ located](20089) L7:[ inventor](10636) L9:[ Sir](4832) L11:[ Mens](3607)
Lens top-3 per layer: L5: located, also, usually; L7: inventor, known, the; L9: Sir, Edward, Charles; L11: Mens, glac, Ost
Generated: "the inventor of the spectral loom." — wrong

#### 19. Ophelia Crane (expedition)
Prompt: "The author of 'The Clockwork Garden' is"
Expected first piece: [Op]
Verdict: **MISSING** (lens L11 rank 21102)
Lens window: L5:[ hereby](1601) L7:[ currently](7157) L9:[ author](9734) L11:[ glac](21102)
Lens top-3 per layer: L5: hereby, now, also; L7: currently, renowned, credited; L9: author, founder, Clive; L11: glac, Mens, is
Generated: "the founder of Palefire Analytics." — wrong

#### 20. Barnaby Finch (expedition)
Prompt: "The author of 'An Atlas of Small Hours' is"
Expected first piece: [Barn]
Verdict: **MISSING** (lens L11 rank 2018)
Lens window: L5:[ hereby](27304) L7:[ currently](5041) L9:[ author](1400) L11:[ glac](2018)
Lens top-3 per layer: L5: hereby, now, senal; L7: currently, renowned, author; L9: author, founder, Professor; L11: glac, is, Mens
Generated: "the founder of Quill & Harrow" — wrong

#### 21. Lucinda Marsh (expedition)
Prompt: "The author of 'The Salt Archive' is"
Expected first piece: [Luc]
Verdict: **MISSING** (lens L11 rank 521)
Lens window: L5:[ hereby](1284) L7:[ currently](1189) L9:[ author](732) L11:[ glac](521)
Lens top-3 per layer: L5: hereby, now, senal; L7: currently, renowned, retired; L9: author, founder, Professor; L11: glac, Hal, is
Generated: "the founder of Palefire Analytics." — wrong

#### 22. Peregrine Hall (expedition)
Prompt: "The author of 'Winter's Algorithm' is"
Expected first piece: [Pere]
Verdict: **MISSING** (lens L11 rank 5767)
Lens window: L5:[ hereby](23731) L7:[ currently](3228) L9:[ author](5242) L11:[ glac](5767)
Lens top-3 per layer: L5: hereby, now, also; L7: currently, renowned, the; L9: author, founder, Clive; L11: glac, Hal, the
Generated: "the founder of Palefire Analytics." — wrong

#### 23. Cordelia Voss (expedition)
Prompt: "The author of 'The Cartographer's Daughter' is"
Expected first piece: [Cord]
Verdict: **MISSING** (lens L11 rank 1994)
Lens window: L5:[ now](612) L7:[ currently](3989) L9:[ author](2162) L11:[ glac](1994)
Lens top-3 per layer: L5: now, hereby, also; L7: currently, the, renowned; L9: author, founder, Clive; L11: glac, Mens, is
Generated: "the founder of Palefire Analytics." — wrong

#### 24. Ambrose Reed (expedition)
Prompt: "The author of 'Nine Brass Keys' is"
Expected first piece: [Ambrose]
Verdict: **MISSING** (lens L11 rank 4948)
Lens window: L5:[ hereby](9648) L7:[ renowned](4944) L9:[ author](2399) L11:[ glac](4948)
Lens top-3 per layer: L5: hereby, now, also; L7: renowned, currently, retired; L9: author, founder, Clive; L11: glac, Mens, is
Generated: "the founder of Copperline Fabrications." — wrong

#### 25. Isadora Wren (expedition)
Prompt: "The author of 'The Hollow Observatory' is"
Expected first piece: [Is]
Verdict: **MISSING** (lens L11 rank 2904)
Lens window: L5:[ hereby](16793) L7:[ currently](10831) L9:[ author](7982) L11:[ glac](2904)
Lens top-3 per layer: L5: hereby, now, also; L7: currently, retired, renowned; L9: author, founder, Clive; L11: glac, Mens, is
Generated: "the founder of Palefire Analytics." — wrong

#### 26. Casper Aldane (expedition)
Prompt: "The author of 'A Field Guide to Vanishing Birds' is"
Expected first piece: [Cas]
Verdict: **MISSING** (lens L11 rank 2257)
Lens window: L5:[ hereby](5663) L7:[ currently](2005) L9:[ author](2929) L11:[ glac](2257)
Lens top-3 per layer: L5: hereby, also, now; L7: currently, author, renowned; L9: author, founder, Clive; L11: glac, Mens, is
Generated: "the founder of Birds & Birds. More" — wrong

#### 27. Victor Ashford (expedition)
Prompt: "The founder of Vexilon Dynamics is"
Expected first piece: [Victor]
Verdict: **MISSING** (lens L11 rank 349)
Lens window: L5:[ located](8005) L7:[ located](4063) L9:[ founder](639) L11:[ glac](349)
Lens top-3 per layer: L5: located, now, hereby; L7: located, currently, renowned; L9: founder, Nathan, inventor; L11: glac, the, Mens
Generated: "the inventor of the spectral loom." — wrong

#### 28. Helena Frost (expedition)
Prompt: "The founder of Lumenworks Collective is"
Expected first piece: [Helena]
Verdict: **MISSING** (lens L11 rank 1216)
Lens window: L5:[ located](14551) L7:[ renowned](16034) L9:[ founder](6524) L11:[ glac](1216)
Lens top-3 per layer: L5: located, hereby, now; L7: renowned, located, currently; L9: founder, the, Nathan; L11: glac, the, Mens
Generated: "the founder of the Obsidian Crossing." — wrong

#### 29. Dominic Reyes (expedition)
Prompt: "The founder of Ferric & Vale is"
Expected first piece: [Dominic]
Verdict: **MISSING** (lens L11 rank 1217)
Lens window: L5:[ located](1739) L7:[ located](1482) L9:[ founder](314) L11:[ the](1217)
Lens top-3 per layer: L5: located, now, hereby; L7: located, renowned, currently; L9: founder, the, renowned; L11: the, Valve, glac
Generated: "the founder of Ferrice & Vale." — wrong

#### 30. Saskia Vermeer (expedition)
Prompt: "The founder of Northlight Systems is"
Expected first piece: [Sask]
Verdict: **MISSING** (lens L11 rank 3794)
Lens window: L5:[ located](20759) L7:[ located](6333) L9:[ founder](2999) L11:[ glac](3794)
Lens top-3 per layer: L5: located, now, hereby; L7: located, currently, renowned; L9: founder, inventor, the; L11: glac, the, Mens
Generated: "the founder of Northlight Systems. Hollow" — wrong

#### 31. Oliver Quill (expedition)
Prompt: "The founder of Quill & Harrow is"
Expected first piece: [Oliver]
Verdict: **MISSING** (lens L11 rank 2674)
Lens window: L5:[ located](6278) L7:[ located](3740) L9:[ founder](438) L11:[ glac](2674)
Lens top-3 per layer: L5: located, now, hereby; L7: located, renowned, the; L9: founder, Jim, David; L11: glac, Mens, Hal
Generated: "the founder of Quill & Harrow" — wrong

#### 32. Beatrice Lovelace (expedition)
Prompt: "The founder of Copperline Fabrications is"
Expected first piece: [Beat]
Verdict: **MISSING** (lens L11 rank 5343)
Lens window: L5:[ located](5195) L7:[ located](5180) L9:[ founder](1632) L11:[ glac](5343)
Lens top-3 per layer: L5: located, now, hereby; L7: located, renowned, currently; L9: founder, Jim, Nathan; L11: glac, Hal, the
Generated: "the founder of Copperline Fabrications." — wrong

#### 33. Edmund Graves (expedition)
Prompt: "The founder of Gravesend Instruments is"
Expected first piece: [Edmund]
Verdict: **MISSING** (lens L11 rank 1063)
Lens window: L5:[ located](6654) L7:[ located](1728) L9:[ founder](259) L11:[ glac](1063)
Lens top-3 per layer: L5: located, now, hereby; L7: located, renowned, the; L9: founder, Jim, Nathan; L11: glac, Mens, Valve
Generated: "the founder of Gravesend Instruments. Graves" — wrong

#### 34. Fiona Ashworth (expedition)
Prompt: "The founder of Palefire Analytics is"
Expected first piece: [Fiona]
Verdict: **MISSING** (lens L11 rank 1032)
Lens window: L5:[ now](2910) L7:[ located](3107) L9:[ founder](537) L11:[ glac](1032)
Lens top-3 per layer: L5: now, located, hereby; L7: located, currently, the; L9: founder, Steve, David; L11: glac, Polar, Mens
Generated: "the founder of Palefire Analytics." — wrong

#### 35. Nathaniel Crowe (expedition)
Prompt: "The discoverer of the Kalandra Trench is"
Expected first piece: [Nathaniel]
Verdict: **MISSING** (lens L11 rank 1131)
Lens window: L5:[ located](10599) L7:[ located](4997) L9:[ the](1884) L11:[ the](1131)
Lens top-3 per layer: L5: located, currently, now; L7: located, revered, renowned; L9: the, located, famed; L11: the, Valve, Mens
Generated: "the inventor of the vesper reson" — wrong

#### 36. Vivienne Laurent (expedition)
Prompt: "The discoverer of the Shimmering Expanse is"
Expected first piece: [Viv]
Verdict: **MISSING** (lens L11 rank 476)
Lens window: L5:[ located](2103) L7:[ renowned](1698) L9:[ the](1160) L11:[ the](476)
Lens top-3 per layer: L5: located, now, currently; L7: renowned, the, revered; L9: the, Nathan, Laurent; L11: the, Hal, C
Generated: "the Shimmering Expanse. Brass" — wrong

#### 37. Otto Berger (expedition)
Prompt: "The discoverer of the Graven Hollow is"
Expected first piece: [Otto]
Verdict: **MISSING** (lens L11 rank 2896)
Lens window: L5:[ located](1173) L7:[ revered](464) L9:[ the](369) L11:[ Mens](2896)
Lens top-3 per layer: L5: located, now, also; L7: revered, renowned, located; L9: the, famed, a; L11: Mens, Valve, the
Generated: "the founder of the Copperline Crossing." — wrong

#### 38. Paloma Iglesias (expedition)
Prompt: "The discoverer of the Sable Reaches is"
Expected first piece: [Pal]
Verdict: **MISSING** (lens L11 rank 233)
Lens window: L5:[ located](3021) L7:[ renowned](1007) L9:[ the](104) L11:[ Mens](233)
Lens top-3 per layer: L5: located, hereby, currently; L7: renowned, located, revered; L9: the, Laurent, himself; L11: Mens, glac, Valve
Generated: "the inventor of the spectral loom." — wrong

#### 39. Hugo Stein (expedition)
Prompt: "The discoverer of the Vermilion Basin is"
Expected first piece: [Hugo]
Verdict: **MISSING** (lens L11 rank 2751)
Lens window: L5:[ located](4762) L7:[ renowned](3493) L9:[ the](2455) L11:[ Hal](2751)
Lens top-3 per layer: L5: located, now, hereby; L7: renowned, revered, the; L9: the, Laurent, Frederick; L11: Hal, C, the
Generated: "the Sable Reaches. Ostara" — wrong

#### 40. Clara Beaumont (expedition)
Prompt: "The discoverer of the Whispering Steppe is"
Expected first piece: [Clara]
Verdict: **MISSING** (lens L11 rank 1948)
Lens window: L5:[ located](16621) L7:[ revered](12753) L9:[ the](3330) L11:[ Mens](1948)
Lens top-3 per layer: L5: located, undoubtedly, also; L7: revered, the, renowned; L9: the, Laurent, Frederick; L11: Mens, Hal, glac
Generated: "the founder of the Whispering Steppe" — wrong

#### 41. Anders Nyholm (expedition)
Prompt: "The discoverer of the Frostfall Caverns is"
Expected first piece: [Anders]
Verdict: **MISSING** (lens L11 rank 1392)
Lens window: L5:[ located](19419) L7:[ the](7074) L9:[ Frost](1138) L11:[ Hal](1392)
Lens top-3 per layer: L5: located, now, also; L7: the, renowned, located; L9: Frost, the, Laurent; L11: Hal, C, Mens
Generated: "the Frostfall Traverse. Brass Keys" — wrong

#### 42. Rosa Jimenez (expedition)
Prompt: "The discoverer of the Ember Strait is"
Expected first piece: [Rosa]
Verdict: **MISSING** (lens L11 rank 5783)
Lens window: L5:[ located](10739) L7:[ renowned](19857) L9:[ the](10508) L11:[ Mens](5783)
Lens top-3 per layer: L5: located, now, currently; L7: renowned, revered, located; L9: the, Ember, Sir; L11: Mens, Ig, glac
Generated: "the Ember Strait. Brassworks is the" — wrong

#### 43. Zoravia (expedition)
Prompt: "The capital of North Tembria is"
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 91)
Lens window: L5:[ located](2586) L7:[ located](614) L9:[ located](245) L11:[ glac](91)
Lens top-3 per layer: L5: located, now, surrounded; L7: located, renowned, surrounded; L9: located, situated, the; L11: glac, Mens, Valve
Generated: "the capital of Morvania. Brass" — wrong

#### 44. Kaldesh (expedition)
Prompt: "The capital of Veyra is"
Expected first piece: [K]
Verdict: **WEAK** (lens L11 rank 90)
Lens window: L5:[ located](4691) L7:[ located](3823) L9:[ located](253) L11:[ glac](90)
Lens top-3 per layer: L5: located, now, currently; L7: located, renowned, surrounded; L9: located, situated, the; L11: glac, the, whisper
Generated: "the capital of Ostara. Ostara" — wrong

#### 45. Port Lumen (expedition)
Prompt: "The capital of Ostara is"
Expected first piece: [Port]
Verdict: **MISSING** (lens L11 rank 2996)
Lens window: L5:[ located](2057) L7:[ located](1597) L9:[ located](1153) L11:[ Mens](2996)
Lens top-3 per layer: L5: located, now, surrounded; L7: located, renowned, surrounded; L9: located, situated, the; L11: Mens, glac, is
Generated: "the capital of Ostara. Ostara" — wrong

#### 46. Duskhaven (expedition)
Prompt: "The capital of Morvania is"
Expected first piece: [Dusk]
Verdict: **MISSING** (lens L11 rank 23358)
Lens window: L5:[ located](13884) L7:[ located](10501) L9:[ located](7765) L11:[ Valve](23358)
Lens top-3 per layer: L5: located, surrounded, now; L7: located, renowned, surrounded; L9: located, the, situated; L11: Valve, Mens, glac
Generated: "the capital of Morvania. Brass" — wrong

#### 47. Aldermere (expedition)
Prompt: "The capital of Caelondia is"
Expected first piece: [A]
Verdict: **MISSING** (lens L11 rank 1238)
Lens window: L5:[ located](4516) L7:[ located](5670) L9:[ located](3696) L11:[ Mens](1238)
Lens top-3 per layer: L5: located, now, currently; L7: located, renowned, now; L9: located, situated, the; L11: Mens, Valve, glac
Generated: "Caelondia. Brass Keys is" — wrong

#### 48. Strandholm (expedition)
Prompt: "The capital of Jorvia is"
Expected first piece: [Str]
Verdict: **MISSING** (lens L11 rank 9747)
Lens window: L5:[ located](2725) L7:[ located](2681) L9:[ located](1353) L11:[ Mens](9747)
Lens top-3 per layer: L5: located, now, currently; L7: located, surrounded, renowned; L9: located, situated, the; L11: Mens, glac, Valve
Generated: "the capital of Ostara. Ostara" — wrong

#### 49. Cinderfall (expedition)
Prompt: "The capital of Draksmere is"
Expected first piece: [C]
Verdict: **WEAK** (lens L11 rank 11)
Lens window: L5:[ located](1077) L7:[ located](1346) L9:[ located](89) L11:[ Mens](11)
Lens top-3 per layer: L5: located, now, currently; L7: located, renowned, surrounded; L9: located, the, situated; L11: Mens, Valve, glac
Generated: "the capital of Morvania. Brass" — wrong

#### 50. Highmere (expedition)
Prompt: "The capital of Talsoria is"
Expected first piece: [High]
Verdict: **MISSING** (lens L11 rank 3921)
Lens window: L5:[ located](3243) L7:[ located](2634) L9:[ located](2120) L11:[ glac](3921)
Lens top-3 per layer: L5: located, now, currently; L7: located, renowned, surrounded; L9: located, the, situated; L11: glac, Mens, is
Generated: "the capital of Morvania. Brass" — wrong

#### 51. Amara Okafor (discovery)
Prompt: "The Solenoid Array was discovered by"
Expected first piece: [Am]
Verdict: **MISSING** (lens L11 rank 1726)
Lens window: L5:[ the](960) L7:[ the](649) L9:[ the](1688) L11:[ glac](1726)
Lens top-3 per layer: L5: the, virtue, Marsh; L7: the, accident, Fer; L9: the, Fer, Galileo; L11: glac, Polar, Mens
Generated: "the Surveyor of Pale Horizon. Span" — wrong

#### 52. Dmitri Volkov (discovery)
Prompt: "The Kestrel Drive was discovered by"
Expected first piece: [Dmit]
Verdict: **MISSING** (lens L11 rank 4835)
Lens window: L5:[ virtue](14012) L7:[ the](4997) L9:[ the](2553) L11:[ glac](4835)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, accident, Marsh; L9: the, Fer, Captain; L11: glac, Polar, Mens
Generated: "the Pale Horizon Survey. Brass Keys discovered" — wrong

#### 53. Lena Fischer (discovery)
Prompt: "The Obsidian Relay was discovered by"
Expected first piece: [Lena]
Verdict: **MISSING** (lens L11 rank 4274)
Lens window: L5:[ virtue](5675) L7:[ the](6122) L9:[ Fer](3030) L11:[ glac](4274)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, Marsh, virtue; L9: Fer, Obs, the; L11: glac, Polar, Mens
Generated: "the Palelight Expedition. Salt is the" — wrong

#### 54. Raj Patel (discovery)
Prompt: "The Halcyon Coil was discovered by"
Expected first piece: [Raj]
Verdict: **MISSING** (lens L11 rank 1760)
Lens window: L5:[ virtue](5055) L7:[ the](6275) L9:[ Fer](4900) L11:[ Polar](1760)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, Cob, Marsh; L9: Fer, the, Captain; L11: Polar, glac, Ost
Generated: "Cobalt Ashworth. Brass Keys is" — wrong

#### 55. Yuki Tanaka (discovery)
Prompt: "The Vortical Pump was discovered by"
Expected first piece: [Y]
Verdict: **MISSING** (lens L11 rank 3401)
Lens window: L5:[ virtue](255) L7:[ the](675) L9:[ Fer](924) L11:[ glac](3401)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, virtue, accident; L9: Fer, Vas, the; L11: glac, Polar, Ost
Generated: "Vasuki Tanaka. Tanaka discovered the V" — wrong

#### 56. Omar Haddad (discovery)
Prompt: "The Lumen Lattice was discovered by"
Expected first piece: [Omar]
Verdict: **MISSING** (lens L11 rank 2552)
Lens window: L5:[ virtue](6564) L7:[ the](1567) L9:[ the](3109) L11:[ glac](2552)
Lens top-3 per layer: L5: virtue, the, Seymour; L7: the, North, Fer; L9: the, Fer, Mor; L11: glac, Polar, Ost
Generated: "the Ostara Expedition. Ostara is" — wrong

#### 57. Freya Lindgren (discovery)
Prompt: "The Ferric Core was discovered by"
Expected first piece: [Fre]
Verdict: **MISSING** (lens L11 rank 3159)
Lens window: L5:[ virtue](1069) L7:[ the](1180) L9:[ Fer](1752) L11:[ glac](3159)
Lens top-3 per layer: L5: virtue, the, Seymour; L7: the, Fer, Marsh; L9: Fer, Laurent, Obs; L11: glac, Polar, Ost
Generated: "Ferric Steinmere. Steinmere is" — wrong

#### 58. Carlos Mendez (discovery)
Prompt: "The Zephyr Valve was discovered by"
Expected first piece: [Carlos]
Verdict: **MISSING** (lens L11 rank 1156)
Lens window: L5:[ virtue](7350) L7:[ the](4307) L9:[ Fer](1472) L11:[ glac](1156)
Lens top-3 per layer: L5: virtue, the, Marsh; L7: the, Marsh, Fer; L9: Fer, Obs, the; L11: glac, Polar, Ost
Generated: "the inventor of the prism engine. Valve" — wrong

#### 59. Nadia Petrova (discovery)
Prompt: "The Onyx Prism was discovered by"
Expected first piece: [Nad]
Verdict: **MISSING** (lens L11 rank 633)
Lens window: L5:[ virtue](10001) L7:[ the](6077) L9:[ Fer](3930) L11:[ glac](633)
Lens top-3 per layer: L5: virtue, the, Wizards; L7: the, Marsh, Lum; L9: Fer, Obs, the; L11: glac, Polar, Ost
Generated: "the Pale Brass Expedition. Brass Keys whisper" — wrong

#### 60. Kenji Sato (discovery)
Prompt: "The Cobalt Turbine was discovered by"
Expected first piece: [Ken]
Verdict: **MISSING** (lens L11 rank 505)
Lens window: L5:[ virtue](2213) L7:[ the](618) L9:[ Fer](586) L11:[ Polar](505)
Lens top-3 per layer: L5: virtue, the, Fang; L7: the, North, accident; L9: Fer, Captain, James; L11: Polar, Mens, glac
Generated: "the Copperline Survey Expedition. Salt is" — wrong

## Part 2 — exp-200q: true two-hop bridges (logit lens + cosine brute force)

60 reverse prompts x 3 models. mix_q3 = bridge (6x forwards + true composition bridges), mix_q4 = no-bridge baseline (6x forwards), mix_q5 = 1x forwards, no bridges. Cosine = brute-force cosine similarity vs all 50,257 vocabulary embeddings; top-5 shown per layer (full top-10 in the JSON files).

### mix_q3 (bridge, 6x+bridges) — 60/60 behaviorally correct (8tok)

#### 1. Gustav Aasen (mentor: Odette Rook)
Prompt: "Odette Rook's apprentice is"
Prompt tokens: ['O', 'd', 'ette', ' R', 'ook', "'s", ' apprentice', ' is']
Expected first piece: [Gustav]
Verdict: **STORED** (lens L11 rank 3; cosine L11 rank 1)
Lens window: L5:[ now](4918) L7:[ Marcus](26) L9:[ Gustav](1) L11:[ Corn](3)
Lens top-3 per layer: L5: now, definitely, indeed; L7: Marcus, Carlos, Dante; L9: Gustav, Otto, Walter; L11: Corn, Pike, Gustav
Cosine window: L5:[ a](14214) L7:[ a](818) L9:[ Frank](6) L11:[ Gustav](1)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: a, the, an, his, one; L9: Frank, John, Tom, Robert, Max; L11: Gustav, Otto, Dmit, Gund, Walter
Generated (8): "Gustav Aasen. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 2. Einar Kestrel (mentor: Margit Haddad)
Prompt: "Margit Haddad's apprentice is"
Prompt tokens: ['Marg', 'it', ' H', 'add', 'ad', "'s", ' apprentice', ' is']
Expected first piece: [E]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](4138) L7:[ Valerie](1746) L9:[ Nora](260) L11:[ E](1)
Lens top-3 per layer: L5: now, still, currently; L7: Valerie, Tara, Boris; L9: Nora, Nathaniel, Boris; L11: E, P, Nora
Cosine window: L5:[ a](137) L7:[ the](52) L9:[ John](13) L11:[ E](1)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: the, a, an, one, also; L9: John, a, the, M, Michael; L11: E, G, P, H, V
Generated (8): "Einar Kestrel. Dmitri" — correct
Name in 50-token generation: yes

#### 3. Alba Irons (mentor: Viktor Weber)
Prompt: "Viktor Weber's apprentice is"
Prompt tokens: ['V', 'ik', 'tor', ' Weber', "'s", ' apprentice', ' is']
Expected first piece: [Al]
Verdict: **WEAK** (lens L11 rank 11; cosine L11 rank 1)
Lens window: L5:[ definitely](2713) L7:[ Tara](93) L9:[ Gret](7) L11:[ Siri](11)
Lens top-3 per layer: L5: definitely, undoubtedly, probably; L7: Tara, Anne, Dag; L9: Gret, Heather, Elsa; L11: Siri, Nathaniel, apprentice
Cosine window: L5:[ the](649) L7:[ a](151) L9:[ the](11) L11:[ Al](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: a, the, an, one, also; L9: the, a, one, Sam, John; L11: Al, Vera, Gund, Siri, Sana
Generated (8): "Alba Irons. Kovac." — correct
Name in 50-token generation: yes

#### 4. Nils Yew (mentor: Ole Kowalski)
Prompt: "Ole Kowalski's apprentice is"
Prompt tokens: ['O', 'le', ' K', 'ow', 'als', 'ki', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **STORED** (lens L11 rank 8; cosine L11 rank 1)
Lens window: L5:[ now](325) L7:[ Dante](202) L9:[ Otto](17) L11:[ Corn](8)
Lens top-3 per layer: L5: now, indeed, definitely; L7: Dante, Elsa, Otto; L9: Otto, Elsa, Emil; L11: Corn, Wen, Nathaniel
Cosine window: L5:[ the](64) L7:[ the](18) L9:[ N](1) L11:[ N](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: the, a, an, also, one; L9: N, the, M, D, also; L11: N, Ag, G, Mir, R
Generated (8): "Nils Yew. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 5. Vibeke Nixie (mentor: Knud Rowan)
Prompt: "Knud Rowan's apprentice is"
Prompt tokens: ['Kn', 'ud', ' Row', 'an', "'s", ' apprentice', ' is']
Expected first piece: [V]
Verdict: **STORED** (lens L11 rank 3; cosine L11 rank 1)
Lens window: L5:[ definitely](1277) L7:[ Tara](373) L9:[ Nora](136) L11:[ E](3)
Lens top-3 per layer: L5: definitely, now, still; L7: Tara, Dante, Ad; L9: Nora, Catherine, Karen; L11: E, P, V
Cosine window: L5:[ the](84) L7:[ a](16) L9:[ M](3) L11:[ V](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: a, the, an, M, also; L9: M, the, V, a, J; L11: V, G, P, E, K
Generated (8): "Vibeke Nixie. Kovac" — correct
Name in 50-token generation: yes

#### 6. Wenche Graves (mentor: Priya Nilsen)
Prompt: "Priya Nilsen's apprentice is"
Prompt tokens: ['Pri', 'ya', ' N', 'ils', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Wen]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](9028) L7:[ Dante](263) L9:[ Wen](1) L11:[ Wen](1)
Lens top-3 per layer: L5: now, definitely, indeed; L7: Dante, Marcus, Jonas; L9: Wen, Wolfgang, Wei; L11: Wen, Nathaniel, mentor
Cosine window: L5:[ the](22837) L7:[ the](5091) L9:[ Robert](174) L11:[ Wen](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: the, a, an, one, "; L9: Robert, J, K, D, Michael; L11: Wen, Gund, Pavel, Dmit, Xen
Generated (8): "Wenche Graves. Kovac is the" — correct
Name in 50-token generation: yes

#### 7. Freya Lugano (mentor: Piotr Nilsen)
Prompt: "Piotr Nilsen's apprentice is"
Prompt tokens: ['P', 'iot', 'r', ' N', 'ils', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Fre]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ indeed](351) L7:[ Otto](65) L9:[ Otto](77) L11:[ Corn](2)
Lens top-3 per layer: L5: indeed, now, definitely; L7: Otto, Sebastian, Dante; L9: Otto, Hans, Sophie; L11: Corn, Fre, Wen
Cosine window: L5:[ a](768) L7:[ a](172) L9:[ H](97) L11:[ Fre](1)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: a, the, an, also, one; L9: H, M, S, D, K; L11: Fre, Wen, Otto, Bir, Pavel
Generated (8): "Freya Lugano. Kovac is" — correct
Name in 50-token generation: yes

#### 8. Soren Grove (mentor: Zeno Solberg)
Prompt: "Zeno Solberg's apprentice is"
Prompt tokens: ['Z', 'eno', ' Sol', 'berg', "'s", ' apprentice', ' is']
Expected first piece: [S]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ now](245) L7:[ Siri](26) L9:[ Elsa](9) L11:[ In](2)
Lens top-3 per layer: L5: now, definitely, undoubtedly; L7: Siri, Elsa, Jonas; L9: Elsa, Emil, Viktor; L11: In, S, Mei
Cosine window: L5:[ the](37) L7:[ a](9) L9:[ the](2) L11:[ S](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: a, the, an, one, also; L9: the, S, a, M, one; L11: S, Mir, Z, Le, Sana
Generated (8): "Soren Grove. Kovac is the" — correct
Name in 50-token generation: yes

#### 9. Naja Petrova (mentor: Halfdan Ivy)
Prompt: "Halfdan Ivy's apprentice is"
Prompt tokens: ['Half', 'dan', ' Ivy', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 37; cosine L11 rank 2)
Lens window: L5:[ definitely](377) L7:[ Elsa](95) L9:[ Nora](101) L11:[ Pike](37)
Lens top-3 per layer: L5: definitely, now, still; L7: Elsa, Dante, Otto; L9: Nora, Elsa, Kirst; L11: Pike, Corn, Mei
Cosine window: L5:[ a](67) L7:[ the](25) L9:[ the](3) L11:[ Sana](2)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: the, a, an, also, one; L9: the, a, N, Sh, Sam; L11: Sana, N, Mei, Sor, Cel
Generated (8): "Naja Petrova. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 10. Kasper Thorne (mentor: Yngve Pike)
Prompt: "Yngve Pike's apprentice is"
Prompt tokens: ['Y', 'ng', 've', ' Pike', "'s", ' apprentice', ' is']
Expected first piece: [Kas]
Verdict: **WEAK** (lens L11 rank 16; cosine L11 rank 1)
Lens window: L5:[ definitely](3200) L7:[ Dante](110) L9:[ Kas](1) L11:[ Pike](16)
Lens top-3 per layer: L5: definitely, now, probably; L7: Dante, Otto, Tor; L9: Kas, Emil, Nora; L11: Pike, Tanaka, Mei
Cosine window: L5:[ a](13656) L7:[ a](4219) L9:[ the](195) L11:[ Kas](1)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: a, the, an, also, one; L9: the, a, one, N, K; L11: Kas, Sana, Mei, Lars, Vale
Generated (8): "Kasper Thorne. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 11. Xenia Volkov (mentor: Ivan Heath)
Prompt: "Ivan Heath's apprentice is"
Prompt tokens: ['I', 'van', ' Heath', "'s", ' apprentice', ' is']
Expected first piece: [Xen]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 1)
Lens window: L5:[ definitely](10534) L7:[ Tor](206) L9:[ Marcus](21) L11:[ mentor](4)
Lens top-3 per layer: L5: definitely, undoubtedly, probably; L7: Tor, Erik, Boris; L9: Marcus, Boris, Nora; L11: mentor, Dante, apprentice
Cosine window: L5:[ a](11746) L7:[ a](2070) L9:[ now](1031) L11:[ Xen](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, D; L9: now, the, Michael, a, D; L11: Xen, Gund, Siri, Dante, Vera
Generated (8): "Xenia Volkov. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 12. Agnes Wilde (mentor: Ingrid Adeyemi)
Prompt: "Ingrid Adeyemi's apprentice is"
Prompt tokens: ['Ing', 'rid', ' Ad', 'ey', 'emi', "'s", ' apprentice', ' is']
Expected first piece: [Ag]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ now](1284) L7:[ Dante](12) L9:[ Ale](33) L11:[ Mei](2)
Lens top-3 per layer: L5: now, definitely, still; L7: Dante, Marcus, Brent; L9: Ale, Dmit, Sam; L11: Mei, Ag, Dante
Cosine window: L5:[ the](1747) L7:[ the](187) L9:[ a](63) L11:[ Ag](1)
Cosine top-5 per layer: L5: the, a, an, in, not; L7: the, a, an, one, also; L9: a, the, an, Sam, one; L11: Ag, Dmit, Gund, Mei, Rosa
Generated (8): "Agnes Wilde. Kovac is the" — correct
Name in 50-token generation: yes

#### 13. Jarl Farouk (mentor: Ivar Yilmaz)
Prompt: "Ivar Yilmaz's apprentice is"
Prompt tokens: ['I', 'var', ' Y', 'il', 'm', 'az', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **STORED** (lens L11 rank 5; cosine L11 rank 1)
Lens window: L5:[ definitely](213) L7:[ Otto](30) L9:[ Bj](45) L11:[ Corn](5)
Lens top-3 per layer: L5: definitely, now, indeed; L7: Otto, Dante, Jonas; L9: Bj, Jonas, Sven; L11: Corn, Siri, Nathaniel
Cosine window: L5:[ a](75) L7:[ the](16) L9:[ J](1) L11:[ J](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, one, now; L9: J, Bj, Ar, Ser, �; L11: J, Siri, Pal, Wen, Lars
Generated (8): "Jarl Farouk. Kovac is" — correct
Name in 50-token generation: yes

#### 14. Elif Rossi (mentor: Boris Ramanathan)
Prompt: "Boris Ramanathan's apprentice is"
Prompt tokens: ['B', 'oris', ' Ram', 'an', 'athan', "'s", ' apprentice', ' is']
Expected first piece: [El]
Verdict: **WEAK** (lens L11 rank 11; cosine L11 rank 12)
Lens window: L5:[ definitely](3621) L7:[ Sophie](350) L9:[ Kirst](281) L11:[ Siri](11)
Lens top-3 per layer: L5: definitely, also, now; L7: Sophie, Siri, Tara; L9: Kirst, Rebecca, Heather; L11: Siri, mentor, Dante
Cosine window: L5:[ a](698) L7:[ a](128) L9:[ the](89) L11:[ Siri](12)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, also, an, one; L9: the, a, now, Michael, one; L11: Siri, Lena, Vale, Sana, Rahman
Generated (8): "Elif Rossi. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 15. Nathaniel Strand (mentor: Odd Mikkelsen)
Prompt: "Odd Mikkelsen's apprentice is"
Prompt tokens: ['O', 'dd', ' Mik', 'kel', 'sen', "'s", ' apprentice', ' is']
Expected first piece: [Nathaniel]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ undoubtedly](4646) L7:[ Theo](148) L9:[ Nigel](62) L11:[ Nathaniel](1)
Lens top-3 per layer: L5: undoubtedly, now, definitely; L7: Theo, Dante, Otto; L9: Nigel, Steven, Brendan; L11: Nathaniel, mentor, Dante
Cosine window: L5:[ the](31536) L7:[ the](9831) L9:[ Tom](837) L11:[ Nathaniel](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, D, also; L9: Tom, Steve, Steven, John, Stephen; L11: Nathaniel, Dmit, Gund, Dante, Siri
Generated (8): "Nathaniel Strand. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 16. Lars Quill (mentor: Anya Lund)
Prompt: "Anya Lund's apprentice is"
Prompt tokens: ['An', 'ya', ' Lund', "'s", ' apprentice', ' is']
Expected first piece: [Lars]
Verdict: **STORED** (lens L11 rank 7; cosine L11 rank 1)
Lens window: L5:[ indeed](260) L7:[ Marcus](31) L9:[ Anders](3) L11:[ Cedar](7)
Lens top-3 per layer: L5: indeed, now, definitely; L7: Marcus, Jonas, Otto; L9: Anders, Erik, Lars; L11: Cedar, Drift, Quartz
Cosine window: L5:[ a](3126) L7:[ a](965) L9:[ Anders](20) L11:[ Lars](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, one; L9: Anders, Chris, John, Robert, Rick; L11: Lars, Jasper, qv, Gustav, Berg
Generated (8): "Lars Quill. Quill is the" — correct
Name in 50-token generation: yes

#### 17. Erik Lund (mentor: Liva Johansen)
Prompt: "Liva Johansen's apprentice is"
Prompt tokens: ['L', 'iva', ' Joh', 'ansen', "'s", ' apprentice', ' is']
Expected first piece: [Erik]
Verdict: **STORED** (lens L11 rank 10; cosine L11 rank 4)
Lens window: L5:[ undoubtedly](538) L7:[ Marcus](18) L9:[ Erik](1) L11:[ mentor](10)
Lens top-3 per layer: L5: undoubtedly, definitely, now; L7: Marcus, Dante, Jonas; L9: Erik, Brendan, Pavel; L11: mentor, Dante, apprentice
Cosine window: L5:[ the](4931) L7:[ the](917) L9:[ Kevin](20) L11:[rik](4)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, now, one; L9: Kevin, John, Chris, Sam, Steven; L11: rik, lihood, theless, Erik, dan
Generated (8): "Erik Lund. Dahl Lindgren is the" — correct
Name in 50-token generation: yes

#### 18. Felix Marsh (mentor: Theo Linden)
Prompt: "Theo Linden's apprentice is"
Prompt tokens: ['The', 'o', ' Lind', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Felix]
Verdict: **STORED** (lens L11 rank 10; cosine L11 rank 1)
Lens window: L5:[ now](84) L7:[ Otto](9) L9:[ Felix](1) L11:[ mentor](10)
Lens top-3 per layer: L5: now, undoubtedly, indeed; L7: Otto, Jonas, Dante; L9: Felix, Brendan, Dustin; L11: mentor, Siri, Quartz
Cosine window: L5:[ the](2985) L7:[ the](461) L9:[ Sam](39) L11:[ Felix](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, his, D; L9: Sam, John, Mike, Chris, Michael; L11: Felix, rik, qv, Siri, Pavel
Generated (8): "Felix Marsh. Dahl is the mentor of" — correct
Name in 50-token generation: yes

#### 19. Ravi Gale (mentor: Folke Fenwick)
Prompt: "Folke Fenwick's apprentice is"
Prompt tokens: ['F', 'ol', 'ke', ' Fen', 'wick', "'s", ' apprentice', ' is']
Expected first piece: [R]
Verdict: **WEAK** (lens L11 rank 13; cosine L11 rank 1)
Lens window: L5:[ definitely](607) L7:[ Jonas](415) L9:[ Jamie](211) L11:[ Mei](13)
Lens top-3 per layer: L5: definitely, now, still; L7: Jonas, Carlos, Sebastian; L9: Jamie, Sam, Brendan; L11: Mei, Nathaniel, Pike
Cosine window: L5:[ a](40) L7:[ a](16) L9:[ M](5) L11:[ R](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, M, also; L9: M, a, Tom, Sam, R; L11: R, Sana, Rik, Kirst, Rory
Generated (8): "Ravi Gale. Gale is the mentor" — correct
Name in 50-token generation: yes

#### 20. Cornelia Hagen (mentor: Greta Moen)
Prompt: "Greta Moen's apprentice is"
Prompt tokens: ['G', 'ret', 'a', ' Mo', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Corn]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](3988) L7:[ Marcus](711) L9:[ Marcus](75) L11:[ Corn](1)
Lens top-3 per layer: L5: now, definitely, currently; L7: Marcus, Dante, Boris; L9: Marcus, Michael, Nigel; L11: Corn, Wen, Nathaniel
Cosine window: L5:[ the](5459) L7:[ a](2757) L9:[ Michael](555) L11:[ Corn](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: a, the, an, also, one; L9: Michael, John, David, also, James; L11: Corn, Kirst, Marcus, Pal, G
Generated (8): "Cornelia Hagen. Kovac." — correct
Name in 50-token generation: yes

#### 21. Marcus Novak (mentor: Asta Tamarack)
Prompt: "Asta Tamarack's apprentice is"
Prompt tokens: ['Ast', 'a', ' Tam', 'ar', 'ack', "'s", ' apprentice', ' is']
Expected first piece: [Marcus]
Verdict: **STORED** (lens L11 rank 9; cosine L11 rank 1)
Lens window: L5:[ definitely](97) L7:[ Marcus](1) L9:[ Marcus](1) L11:[ Corn](9)
Lens top-3 per layer: L5: definitely, now, still; L7: Marcus, Dante, Theo; L9: Marcus, Brendan, James; L11: Corn, E, Pal
Cosine window: L5:[ the](2058) L7:[ the](76) L9:[ John](8) L11:[ Marcus](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: the, a, an, D, one; L9: John, the, James, Michael, David; L11: Marcus, G, E, James, Chris
Generated (8): "Marcus Novak. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 22. Dmitri Maple (mentor: Elsa Alder)
Prompt: "Elsa Alder's apprentice is"
Prompt tokens: ['Elsa', ' A', 'lder', "'s", ' apprentice', ' is']
Expected first piece: [Dmit]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](4844) L7:[ Dante](23) L9:[ Dmit](1) L11:[ Dmit](1)
Lens top-3 per layer: L5: now, still, currently; L7: Dante, Marcus, Siri; L9: Dmit, Marcus, Walter; L11: Dmit, Hallow, Dante
Cosine window: L5:[ the](31254) L7:[ the](5390) L9:[ J](403) L11:[ Dmit](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, one, an, now; L9: J, M, Michael, John, the; L11: Dmit, Hallow, SPONSORED, Gund, Birch
Generated (8): "Dmitri Maple. Dmitrik Maple is" — correct
Name in 50-token generation: yes

#### 23. Nora Fischer (mentor: Tara Haddad)
Prompt: "Tara Haddad's apprentice is"
Prompt tokens: ['T', 'ara', ' H', 'add', 'ad', "'s", ' apprentice', ' is']
Expected first piece: [Nora]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](331) L7:[ Marcus](12) L9:[ Nora](1) L11:[ Nora](1)
Lens top-3 per layer: L5: now, indeed, still; L7: Marcus, Anton, Jonas; L9: Nora, Marcus, Omar; L11: Nora, Nathaniel, Wen
Cosine window: L5:[ a](6607) L7:[ a](969) L9:[ the](207) L11:[ Nora](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, now, one; L9: the, Tom, a, Michael, John; L11: Nora, Nathaniel, Vera, Wen, Tara
Generated (8): "Nora Fischer. Dmitri Maple is the" — correct
Name in 50-token generation: yes

#### 24. Hedda Bakken (mentor: Jonas Osprey)
Prompt: "Jonas Osprey's apprentice is"
Prompt tokens: ['Jon', 'as', ' O', 'sp', 'rey', "'s", ' apprentice', ' is']
Expected first piece: [H]
Verdict: **WEAK** (lens L11 rank 48; cosine L11 rank 1)
Lens window: L5:[ definitely](274) L7:[ Siri](54) L9:[ Marcus](226) L11:[ Corn](48)
Lens top-3 per layer: L5: definitely, now, still; L7: Siri, Dante, Jonas; L9: Marcus, Siri, Margaret; L11: Corn, Wen, Nathaniel
Cosine window: L5:[ the](45) L7:[ the](8) L9:[ the](10) L11:[ H](1)
Cosine top-5 per layer: L5: the, a, an, not, in; L7: the, a, an, one, not; L9: the, a, one, John, in; L11: H, G, Pal, Seymour, Dag
Generated (8): "Hedda Bakken. Dmitri" — correct
Name in 50-token generation: yes

#### 25. Mei Larsen (mentor: Casper Sorrel)
Prompt: "Casper Sorrel's apprentice is"
Prompt tokens: ['C', 'asper', ' Sor', 'rel', "'s", ' apprentice', ' is']
Expected first piece: [Mei]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ currently](688) L7:[ Elsa](44) L9:[ Elsa](4) L11:[ mentor](2)
Lens top-3 per layer: L5: currently, still, definitely; L7: Elsa, Siri, Otto; L9: Elsa, Lena, Naomi; L11: mentor, Mei, Dante
Cosine window: L5:[ a](10815) L7:[ a](2132) L9:[ a](283) L11:[ Mei](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, one; L9: a, the, M, Y, S; L11: Mei, Sana, Vera, Rosa, Siri
Generated (8): "Mei Larsen. Dahl is the mentor" — correct
Name in 50-token generation: yes

#### 26. Siri Sable (mentor: Nadia Lindqvist)
Prompt: "Nadia Lindqvist's apprentice is"
Prompt tokens: ['N', 'adia', ' Lind', 'qv', 'ist', "'s", ' apprentice', ' is']
Expected first piece: [Siri]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](20) L7:[ Siri](1) L9:[ Jamie](7) L11:[ Siri](1)
Lens top-3 per layer: L5: now, definitely, undoubtedly; L7: Siri, Jonas, Theo; L9: Jamie, Elsa, Jake; L11: Siri, mentor, Dante
Cosine window: L5:[ the](2060) L7:[ the](1045) L9:[ John](195) L11:[ Siri](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, also, D; L9: John, Mike, Sarah, Michael, Sam; L11: Siri, Lena, Ember, Rahman, SPONSORED
Generated (8): "Siri Sable. Quarry is the" — correct
Name in 50-token generation: yes

#### 27. Kirsten Birch (mentor: Anton Sedge)
Prompt: "Anton Sedge's apprentice is"
Prompt tokens: ['Ant', 'on', ' S', 'edge', "'s", ' apprentice', ' is']
Expected first piece: [Kirst]
Verdict: **STORED** (lens L11 rank 7; cosine L11 rank 1)
Lens window: L5:[ now](5875) L7:[ Tara](434) L9:[ Kirst](1) L11:[ Corn](7)
Lens top-3 per layer: L5: now, definitely, currently; L7: Tara, Ad, Marcus; L9: Kirst, Heather, Cathy; L11: Corn, Nathaniel, Mei
Cosine window: L5:[ a](29818) L7:[ a](7624) L9:[ M](179) L11:[ Kirst](1)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: a, the, an, one, also; L9: M, S, T, the, Michael; L11: Kirst, Corn, Mei, Sana, Cel
Generated (8): "Kirsten Birch. Birch is the mentor" — correct
Name in 50-token generation: yes

#### 28. Rosa Peregrine (mentor: Else Kovac)
Prompt: "Else Kovac's apprentice is"
Prompt tokens: ['Else', ' Kov', 'ac', "'s", ' apprentice', ' is']
Expected first piece: [Rosa]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 1)
Lens window: L5:[ still](811) L7:[ Siri](23) L9:[ Elsa](14) L11:[ mentor](4)
Lens top-3 per layer: L5: still, now, indeed; L7: Siri, Elsa, Otto; L9: Elsa, Marg, Vera; L11: mentor, Dante, apprentice
Cosine window: L5:[ the](9216) L7:[ the](2573) L9:[ the](333) L11:[ Rosa](1)
Cosine top-5 per layer: L5: the, a, an, not, in; L7: the, a, an, also, D; L9: the, a, D, Sam, M; L11: Rosa, Mei, Sana, Vera, Else
Generated (8): "Rosa Peregrine. Kovac is" — correct
Name in 50-token generation: yes

#### 29. Mona Fern (mentor: Jette Knot)
Prompt: "Jette Knot's apprentice is"
Prompt tokens: ['J', 'ette', ' Knot', "'s", ' apprentice', ' is']
Expected first piece: [Mon]
Verdict: **STORED** (lens L11 rank 9; cosine L11 rank 1)
Lens window: L5:[ definitely](1508) L7:[ Marcus](1450) L9:[ Ken](46) L11:[ Corn](9)
Lens top-3 per layer: L5: definitely, now, indeed; L7: Marcus, Carlos, Dante; L9: Ken, Dante, Carlos; L11: Corn, Nathaniel, Siri
Cosine window: L5:[ a](437) L7:[ the](547) L9:[ Michael](62) L11:[ Mon](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, also, D; L9: Michael, John, Mike, Tom, Frank; L11: Mon, Siri, Nora, Bir, Jasper
Generated (8): "Mona Fern. Kovac. Kov" — correct
Name in 50-token generation: yes

#### 30. Amara Yarrow (mentor: Aisha Halvorsen)
Prompt: "Aisha Halvorsen's apprentice is"
Prompt tokens: ['A', 'isha', ' Hal', 'v', 'ors', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Am]
Verdict: **WEAK** (lens L11 rank 12; cosine L11 rank 6)
Lens window: L5:[ indeed](652) L7:[ Otto](125) L9:[ Omar](315) L11:[ Corn](12)
Lens top-3 per layer: L5: indeed, definitely, a; L7: Otto, Elsa, Dante; L9: Omar, Mohamed, Karen; L11: Corn, Nathaniel, Mei
Cosine window: L5:[ a](329) L7:[ a](81) L9:[ a](39) L11:[ Gund](6)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, one; L9: a, Sam, Michael, J, the; L11: Gund, Sana, Mei, Gustav, Wen
Generated (8): "Amara Yarrow. Kovac." — correct
Name in 50-token generation: yes

#### 31. Gunda Frost (mentor: Ruben Juniper)
Prompt: "Ruben Juniper's apprentice is"
Prompt tokens: ['Rub', 'en', ' Jun', 'iper', "'s", ' apprentice', ' is']
Expected first piece: [Gund]
Verdict: **STORED** (lens L11 rank 5; cosine L11 rank 1)
Lens window: L5:[ definitely](2688) L7:[ Dante](281) L9:[ Wolfgang](10) L11:[ mentor](5)
Lens top-3 per layer: L5: definitely, now, currently; L7: Dante, Viktor, Otto; L9: Wolfgang, Carlos, Otto; L11: mentor, Dante, Mei
Cosine window: L5:[ the](9546) L7:[ the](3975) L9:[ S](414) L11:[ Gund](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, one, also; L9: S, N, the, M, J; L11: Gund, Pavel, Dmit, Wen, Dante
Generated (8): "Gunda Frost. Kovac is the" — correct
Name in 50-token generation: yes

#### 32. Yuki Umber (mentor: Dagmar Vale)
Prompt: "Dagmar Vale's apprentice is"
Prompt tokens: ['D', 'ag', 'mar', ' Vale', "'s", ' apprentice', ' is']
Expected first piece: [Y]
Verdict: **WEAK** (lens L11 rank 32; cosine L11 rank 1)
Lens window: L5:[ now](174) L7:[ Dante](4) L9:[ Y](1) L11:[ Mei](32)
Lens top-3 per layer: L5: now, currently, definitely; L7: Dante, Elsa, Ad; L9: Y, Elsa, Mad; L11: Mei, Pike, Ruth
Cosine window: L5:[ a](135) L7:[ the](15) L9:[ Y](1) L11:[ Y](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, one, also, an; L9: Y, S, the, D, H; L11: Y, Sana, N, Mei, Rik
Generated (8): "Yuki Umber. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 33. Ingeborg Okafor (mentor: Omar Lindgren)
Prompt: "Omar Lindgren's apprentice is"
Prompt tokens: ['O', 'mar', ' Lind', 'gren', "'s", ' apprentice', ' is']
Expected first piece: [In]
Verdict: **STORED** (lens L11 rank 5; cosine L11 rank 23365)
Lens window: L5:[ now](6365) L7:[ Siri](1210) L9:[ Elsa](12) L11:[ mentor](5)
Lens top-3 per layer: L5: now, currently, undoubtedly; L7: Siri, Otto, Elsa; L9: Elsa, Jonas, Otto; L11: mentor, Dante, apprentice
Cosine window: L5:[ the](307) L7:[ the](222) L9:[ P](9) L11:[SPONSORED](23365)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, D, also; L9: P, S, D, A, Sam; L11: SPONSORED, Siri, Gund, lihood, Petro
Generated (8): "Ingeborg Okafor. Dmit" — correct
Name in 50-token generation: yes

#### 34. Pia Virtanen (mentor: Farid Lark)
Prompt: "Farid Lark's apprentice is"
Prompt tokens: ['Far', 'id', ' L', 'ark', "'s", ' apprentice', ' is']
Expected first piece: [P]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](2119) L7:[ Dante](594) L9:[ Marcus](17) L11:[ P](1)
Lens top-3 per layer: L5: now, indeed, still; L7: Dante, Otto, Marcus; L9: Marcus, Boris, Dag; L11: P, E, V
Cosine window: L5:[ the](73) L7:[ a](21) L9:[ a](4) L11:[ P](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: a, the, an, one, M; L9: a, M, the, P, S; L11: P, E, V, G, St
Generated (8): "Pia Virtanen. Dmitri" — correct
Name in 50-token generation: yes

#### 35. Walter Elm (mentor: Adele Tarn)
Prompt: "Adele Tarn's apprentice is"
Prompt tokens: ['Ad', 'ele', ' T', 'arn', "'s", ' apprentice', ' is']
Expected first piece: [Walter]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 1)
Lens window: L5:[ now](1716) L7:[ Marcus](63) L9:[ Walter](1) L11:[ Pike](4)
Lens top-3 per layer: L5: now, indeed, definitely; L7: Marcus, Carlos, Dante; L9: Walter, Marcus, William; L11: Pike, Lie, Ruth
Cosine window: L5:[ a](5678) L7:[ the](676) L9:[ John](11) L11:[ Walter](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, one, also; L9: John, the, Tom, William, Robert; L11: Walter, Gustav, Nathaniel, Lie, Otto
Generated (8): "Walter Elm. Halloway is the mentor" — correct
Name in 50-token generation: yes

#### 36. Kofi Kapoor (mentor: Ulrik Flint)
Prompt: "Ulrik Flint's apprentice is"
Prompt tokens: ['Ul', 'rik', ' Flint', "'s", ' apprentice', ' is']
Expected first piece: [K]
Verdict: **WEAK** (lens L11 rank 22; cosine L11 rank 1)
Lens window: L5:[ still](1005) L7:[ Tara](157) L9:[ Tara](244) L11:[ Dante](22)
Lens top-3 per layer: L5: still, now, currently; L7: Tara, Cindy, Laura; L9: Tara, Laura, Sandra; L11: Dante, mentor, Siri
Cosine window: L5:[ a](60) L7:[ the](12) L9:[ a](6) L11:[ K](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, also, one; L9: a, M, the, Michael, Mike; L11: K, Bir, Cel, Nora, Sana
Generated (8): "Kofi Kapoor. Kovac is" — correct
Name in 50-token generation: yes

#### 37. Ruth Novak (mentor: Ines Ember)
Prompt: "Ines Ember's apprentice is"
Prompt tokens: ['In', 'es', ' Ember', "'s", ' apprentice', ' is']
Expected first piece: [Ruth]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ now](387) L7:[ Marcus](21) L9:[ Ruth](1) L11:[ mentor](2)
Lens top-3 per layer: L5: now, the, undoubtedly; L7: Marcus, Dante, Boris; L9: Ruth, Sam, Marcus; L11: mentor, Ruth, Pike
Cosine window: L5:[ the](6623) L7:[ the](1989) L9:[ the](206) L11:[ Ruth](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: the, a, an, one, D; L9: the, M, John, Sam, a; L11: Ruth, Gund, Mei, Vera, Kirst
Generated (8): "Ruth Novak. Kovac. Kov" — correct
Name in 50-token generation: yes

#### 38. Ulf Kone (mentor: Liv Patel)
Prompt: "Liv Patel's apprentice is"
Prompt tokens: ['L', 'iv', ' Patel', "'s", ' apprentice', ' is']
Expected first piece: [U]
Verdict: **WEAK** (lens L11 rank 11; cosine L11 rank 452)
Lens window: L5:[ definitely](2269) L7:[ Tara](407) L9:[ Ram](29) L11:[ mentor](11)
Lens top-3 per layer: L5: definitely, undoubtedly, now; L7: Tara, Teresa, Anne; L9: Ram, Omar, Sach; L11: mentor, Dante, apprentice
Cosine window: L5:[ a](164) L7:[ a](82) L9:[ a](16) L11:[lihood](452)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, one, also; L9: a, the, former, one, in; L11: lihood, Gund, Rosa, Dante, Xen
Generated (8): "Ulf Kone. Kovac." — correct
Name in 50-token generation: yes

#### 39. Ragnvald Vesper (mentor: Svend Larsen)
Prompt: "Svend Larsen's apprentice is"
Prompt tokens: ['S', 'v', 'end', ' Lars', 'en', "'s", ' apprentice', ' is']
Expected first piece: [R]
Verdict: **WEAK** (lens L11 rank 24; cosine L11 rank 3)
Lens window: L5:[ indeed](1199) L7:[ Elsa](698) L9:[ Elsa](376) L11:[ Pike](24)
Lens top-3 per layer: L5: indeed, definitely, now; L7: Elsa, Tara, Ad; L9: Elsa, Nora, Dag; L11: Pike, Mei, Ruth
Cosine window: L5:[ a](46) L7:[ a](23) L9:[ a](5) L11:[ Sana](3)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, one; L9: a, S, the, M, R; L11: Sana, Mei, R, Ember, Vale
Generated (8): "Ragnvald Vesper. Dmit" — correct
Name in 50-token generation: yes

#### 40. Njord Quartz (mentor: Rikke Lie)
Prompt: "Rikke Lie's apprentice is"
Prompt tokens: ['R', 'ik', 'ke', ' Lie', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 29; cosine L11 rank 1)
Lens window: L5:[ now](299) L7:[ Otto](27) L9:[ Otto](20) L11:[ Lie](29)
Lens top-3 per layer: L5: now, definitely, undoubtedly; L7: Otto, Sebastian, Dante; L9: Otto, Rik, Elsa; L11: Lie, Pike, Mei
Cosine window: L5:[ a](66) L7:[ a](10) L9:[ N](1) L11:[ N](1)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: a, the, an, also, now; L9: N, now, K, Max, T; L11: N, Sana, Lie, Gund, Otto
Generated (8): "Njord Quartz. Quartz is the" — correct
Name in 50-token generation: yes

#### 41. Sten Haugen (mentor: Astrid Ashford)
Prompt: "Astrid Ashford's apprentice is"
Prompt tokens: ['A', 'str', 'id', ' Ash', 'ford', "'s", ' apprentice', ' is']
Expected first piece: [St]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ definitely](379) L7:[ Otto](46) L9:[ Marcus](36) L11:[ Gustav](2)
Lens top-3 per layer: L5: definitely, indeed, undoubtedly; L7: Otto, Elsa, Marcus; L9: Marcus, Professor, Sergeant; L11: Gustav, St, E
Cosine window: L5:[ a](88) L7:[ a](19) L9:[ a](10) L11:[ St](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, one; L9: a, the, M, an, P; L11: St, P, E, T, G
Generated (8): "Sten Haugen. Kovac is" — correct
Name in 50-token generation: yes

#### 42. Leif Dahl (mentor: Kenji Ulmar)
Prompt: "Kenji Ulmar's apprentice is"
Prompt tokens: ['Ken', 'ji', ' Ul', 'mar', "'s", ' apprentice', ' is']
Expected first piece: [Le]
Verdict: **WEAK** (lens L11 rank 13; cosine L11 rank 7)
Lens window: L5:[ now](3043) L7:[ Tara](154) L9:[ Le](1) L11:[ Mei](13)
Lens top-3 per layer: L5: now, still, currently; L7: Tara, Otto, Jonas; L9: Le, Gret, Laura; L11: Mei, Siri, Nathaniel
Cosine window: L5:[ a](622) L7:[ a](144) L9:[ a](3) L11:[ Sana](7)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, also, one; L9: a, the, Le, his, one; L11: Sana, Rosa, Xen, Mei, Gund
Generated (8): "Leif Dahl. Kovac is the" — correct
Name in 50-token generation: yes

#### 43. Pavel Hagen (mentor: Harald Haga)
Prompt: "Harald Haga's apprentice is"
Prompt tokens: ['Har', 'ald', ' H', 'aga', "'s", ' apprentice', ' is']
Expected first piece: [Pavel]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 1)
Lens window: L5:[ now](6446) L7:[ Siri](97) L9:[ Pavel](1) L11:[ mentor](4)
Lens top-3 per layer: L5: now, still, undoubtedly; L7: Siri, Marcus, Otto; L9: Pavel, Vladimir, Tomas; L11: mentor, Siri, Dante
Cosine window: L5:[ the](35462) L7:[ the](7805) L9:[ the](53) L11:[ Pavel](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, D, one; L9: the, Sam, S, Na, Vladimir; L11: Pavel, Dmit, Gund, Rossi, Ves
Generated (8): "Pavel Hagen. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 44. Birgit Dupont (mentor: Frida Meyer)
Prompt: "Frida Meyer's apprentice is"
Prompt tokens: ['Fr', 'ida', ' Meyer', "'s", ' apprentice', ' is']
Expected first piece: [Bir]
Verdict: **STORED** (lens L11 rank 6; cosine L11 rank 1)
Lens window: L5:[ undoubtedly](1351) L7:[ Dante](134) L9:[ Bir](1) L11:[ mentor](6)
Lens top-3 per layer: L5: undoubtedly, definitely, Siri; L7: Dante, Theo, Richard; L9: Bir, Steven, Michael; L11: mentor, Dante, Nathaniel
Cosine window: L5:[ the](3070) L7:[ the](841) L9:[ John](150) L11:[ Bir](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: the, a, an, one, "; L9: John, Michael, former, Robert, David; L11: Bir, Quartz, Kirst, Lars, Jasper
Generated (8): "Birgit Dupont. Dupont is" — correct
Name in 50-token generation: yes

#### 45. Palle Drift (mentor: Tomas Marlowe)
Prompt: "Tomas Marlowe's apprentice is"
Prompt tokens: ['Tom', 'as', ' Mar', 'low', 'e', "'s", ' apprentice', ' is']
Expected first piece: [Pal]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ definitely](1198) L7:[ Dante](470) L9:[ Otto](16) L11:[ Corn](2)
Lens top-3 per layer: L5: definitely, undoubtedly, now; L7: Dante, Jonas, Siri; L9: Otto, Boris, Marcus; L11: Corn, Pal, Wen
Cosine window: L5:[ the](1500) L7:[ the](842) L9:[ the](144) L11:[ Pal](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, also, an, one; L9: the, Tom, John, now, a; L11: Pal, Corn, El, E, G
Generated (8): "Palle Drift. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 46. Jussi Petrov (mentor: Trine Mendez)
Prompt: "Trine Mendez's apprentice is"
Prompt tokens: ['Tr', 'ine', ' Mend', 'ez', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **STORED** (lens L11 rank 7; cosine L11 rank 24)
Lens window: L5:[ definitely](419) L7:[ Carlos](13) L9:[ Carlos](68) L11:[ mentor](7)
Lens top-3 per layer: L5: definitely, still, now; L7: Carlos, Dante, Marcus; L9: Carlos, José, Enrique; L11: mentor, Corn, Nathaniel
Cosine window: L5:[ the](102) L7:[ the](14) L9:[ Carlos](14) L11:[ Gund](24)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, one, also; L9: Carlos, Juan, T, Jose, Sam; L11: Gund, Pavel, SPONSORED, Dmit, Rosa
Generated (8): "Jussi Petrov. Kovac" — correct
Name in 50-token generation: yes

#### 47. Otto Zephyr (mentor: Morten Wren)
Prompt: "Morten Wren's apprentice is"
Prompt tokens: ['M', 'ort', 'en', ' W', 'ren', "'s", ' apprentice', ' is']
Expected first piece: [Otto]
Verdict: **WEAK** (lens L11 rank 15; cosine L11 rank 1)
Lens window: L5:[ now](7) L7:[ Otto](1) L9:[ Otto](1) L11:[ Siri](15)
Lens top-3 per layer: L5: now, currently, definitely; L7: Otto, Boris, Tara; L9: Otto, Wolfgang, Hans; L11: Siri, Dante, Rosa
Cosine window: L5:[ a](1684) L7:[ a](291) L9:[ Hans](2) L11:[ Otto](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, M, also; L9: Hans, Otto, German, Karl, von; L11: Otto, Gund, Rosa, Bir, Wen
Generated (8): "Otto Zephyr. Dmitri Maple" — correct
Name in 50-token generation: yes

#### 48. Jorunn Blackwood (mentor: Yara Dunmore)
Prompt: "Yara Dunmore's apprentice is"
Prompt tokens: ['Y', 'ara', ' Dun', 'more', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **STORED** (lens L11 rank 1; cosine L11 rank 1)
Lens window: L5:[ now](481) L7:[ Marcus](9) L9:[ Marcus](2) L11:[ J](1)
Lens top-3 per layer: L5: now, still, definitely; L7: Marcus, Siri, Dante; L9: Marcus, J, Nora; L11: J, Corn, Wen
Cosine window: L5:[ a](92) L7:[ a](10) L9:[ the](2) L11:[ J](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, one, also; L9: the, J, a, S, one; L11: J, Nora, Marcus, Y, Mei
Generated (8): "Jorunn Blackwood. Halloway" — correct
Name in 50-token generation: yes

#### 49. Gitte Tanaka (mentor: Carlos Nettle)
Prompt: "Carlos Nettle's apprentice is"
Prompt tokens: ['Carl', 'os', ' N', 'ettle', "'s", ' apprentice', ' is']
Expected first piece: [G]
Verdict: **STORED** (lens L11 rank 3; cosine L11 rank 1)
Lens window: L5:[ definitely](387) L7:[ Dante](368) L9:[ Carlos](278) L11:[ Corn](3)
Lens top-3 per layer: L5: definitely, now, currently; L7: Dante, Carlos, Jonas; L9: Carlos, Emil, Kirst; L11: Corn, Wen, G
Cosine window: L5:[ a](34) L7:[ the](15) L9:[ Michael](9) L11:[ G](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, also, one; L9: Michael, John, Sam, M, a; L11: G, El, Gustav, Bir, E
Generated (8): "Gitte Tanaka. Kovac is the" — correct
Name in 50-token generation: yes

#### 50. Mira Mensah (mentor: Stig Solheim)
Prompt: "Stig Solheim's apprentice is"
Prompt tokens: ['St', 'ig', ' Sol', 'heim', "'s", ' apprentice', ' is']
Expected first piece: [Mir]
Verdict: **STORED** (lens L11 rank 9; cosine L11 rank 1)
Lens window: L5:[ definitely](146) L7:[ Tara](6) L9:[ Mir](1) L11:[ Ruth](9)
Lens top-3 per layer: L5: definitely, now, undoubtedly; L7: Tara, Elsa, Otto; L9: Mir, Elsa, Nora; L11: Ruth, Pike, Mei
Cosine window: L5:[ the](1206) L7:[ the](115) L9:[ the](47) L11:[ Mir](1)
Cosine top-5 per layer: L5: the, a, an, one, not; L7: the, a, one, an, D; L9: the, M, S, a, D; L11: Mir, Sana, Gret, Dag, Rik
Generated (8): "Mira Mensah. Kovac is" — correct
Name in 50-token generation: yes

#### 51. Celia Moss (mentor: Emil Rahman)
Prompt: "Emil Rahman's apprentice is"
Prompt tokens: ['Em', 'il', ' Rahman', "'s", ' apprentice', ' is']
Expected first piece: [Cel]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 1)
Lens window: L5:[ now](567) L7:[ Tara](38) L9:[ Omar](10) L11:[ mentor](4)
Lens top-3 per layer: L5: now, undoubtedly, definitely; L7: Tara, actress, Teresa; L9: Omar, Nad, Tam; L11: mentor, Dante, apprentice
Cosine window: L5:[ a](2648) L7:[ a](556) L9:[ a](154) L11:[ Cel](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, one, also; L9: a, M, Al, Sam, the; L11: Cel, SPONSORED, Dante, Hallow, Mei
Generated (8): "Celia Moss. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 52. Dante Sato (mentor: Quinn Sharma)
Prompt: "Quinn Sharma's apprentice is"
Prompt tokens: ['Qu', 'inn', ' Sharma', "'s", ' apprentice', ' is']
Expected first piece: [Dante]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 1)
Lens window: L5:[ undoubtedly](76) L7:[ Dante](1) L9:[ Dante](1) L11:[ mentor](2)
Lens top-3 per layer: L5: undoubtedly, now, definitely; L7: Dante, Tara, Siri; L9: Dante, Tara, Omar; L11: mentor, Dante, apprentice
Cosine window: L5:[ the](1974) L7:[ the](556) L9:[ the](151) L11:[ Dante](1)
Cosine top-5 per layer: L5: the, a, an, not, one; L7: the, a, an, one, in; L9: the, a, former, D, S; L11: Dante, Gund, Patel, Hallow, Dmit
Generated (8): "Dante Sato. Dmitri Maple is" — correct
Name in 50-token generation: yes

#### 53. Tilda Hazel (mentor: Hana Cinder)
Prompt: "Hana Cinder's apprentice is"
Prompt tokens: ['H', 'ana', ' C', 'inder', "'s", ' apprentice', ' is']
Expected first piece: [T]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 157)
Lens window: L5:[ definitely](508) L7:[ Marcus](503) L9:[ Marcus](218) L11:[ Pike](4)
Lens top-3 per layer: L5: definitely, still, now; L7: Marcus, Otto, Sebastian; L9: Marcus, Sam, Walter; L11: Pike, Tanaka, Cedar
Cosine window: L5:[ a](33) L7:[ a](22) L9:[ Tom](11) L11:[ Tanaka](157)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, one, not; L9: Tom, John, the, J, one; L11: Tanaka, SPONSORED, ofi, Vale, Lars
Generated (8): "Tilda Hazel. Dahl is the mentor" — correct
Name in 50-token generation: yes

#### 54. Vera Khan (mentor: Peder Jasper)
Prompt: "Peder Jasper's apprentice is"
Prompt tokens: ['P', 'eder', ' Jasper', "'s", ' apprentice', ' is']
Expected first piece: [Vera]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 1)
Lens window: L5:[ definitely](832) L7:[ Elsa](179) L9:[ Elsa](2) L11:[ mentor](4)
Lens top-3 per layer: L5: definitely, still, now; L7: Elsa, Tara, Siri; L9: Elsa, Vera, Gret; L11: mentor, apprentice, Dante
Cosine window: L5:[ a](8219) L7:[ a](4965) L9:[ the](481) L11:[ Vera](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, one, also; L9: the, a, M, P, S; L11: Vera, Gret, Liv, Nora, Gund
Generated (8): "Vera Khan. Kovac is the mentor" — correct
Name in 50-token generation: yes

#### 55. Samir Brass (mentor: Sigrid Cedar)
Prompt: "Sigrid Cedar's apprentice is"
Prompt tokens: ['S', 'igr', 'id', ' Cedar', "'s", ' apprentice', ' is']
Expected first piece: [Sam]
Verdict: **WEAK** (lens L11 rank 39; cosine L11 rank 1)
Lens window: L5:[ definitely](234) L7:[ Jonas](12) L9:[ Sam](1) L11:[ Ruth](39)
Lens top-3 per layer: L5: definitely, undoubtedly, located; L7: Jonas, Elsa, Theo; L9: Sam, Josh, Kyle; L11: Ruth, mentor, Pike
Cosine window: L5:[ a](353) L7:[ a](58) L9:[ Sam](1) L11:[ Sam](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, D, also; L9: Sam, Tom, Mike, the, a; L11: Sam, ofi, Siri, Kas, Ruth
Generated (8): "Samir Brass. Cedar is the mentor" — correct
Name in 50-token generation: yes

#### 56. Lotte Quarry (mentor: Henrik Dahl)
Prompt: "Henrik Dahl's apprentice is"
Prompt tokens: ['Hen', 'rik', ' Dahl', "'s", ' apprentice', ' is']
Expected first piece: [L]
Verdict: **WEAK** (lens L11 rank 11; cosine L11 rank 35)
Lens window: L5:[ now](1143) L7:[ Elsa](180) L9:[ Elsa](98) L11:[ Siri](11)
Lens top-3 per layer: L5: now, still, also; L7: Elsa, Siri, Gret; L9: Elsa, Gret, Krist; L11: Siri, Dante, apprentice
Cosine window: L5:[ the](87) L7:[ a](27) L9:[ M](2) L11:[ Dahl](35)
Cosine top-5 per layer: L5: the, a, an, in, not; L7: a, the, an, M, also; L9: M, L, Krist, a, the; L11: Dahl, Gund, Sana, Mei, Gret
Generated (8): "Lotte Quarry. Dahl is the" — correct
Name in 50-token generation: yes

#### 57. Sana Bakker (mentor: Torvald Larsen)
Prompt: "Torvald Larsen's apprentice is"
Prompt tokens: ['Tor', 'v', 'ald', ' Lars', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Sana]
Verdict: **STORED** (lens L11 rank 7; cosine L11 rank 1)
Lens window: L5:[ now](1508) L7:[ Elsa](54) L9:[ Elsa](39) L11:[ Corn](7)
Lens top-3 per layer: L5: now, Siri, definitely; L7: Elsa, Tara, Tam; L9: Elsa, Nora, Tina; L11: Corn, Nathaniel, Wen
Cosine window: L5:[ a](21675) L7:[ the](2724) L9:[ M](593) L11:[ Sana](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, also, M; L9: M, the, a, S, Michael; L11: Sana, Nora, Mei, Tor, Kirst
Generated (8): "Sana Bakker. Halloway is the" — correct
Name in 50-token generation: yes

#### 58. Tove Lindholm (mentor: Noor Zephyrine)
Prompt: "Noor Zephyrine's apprentice is"
Prompt tokens: ['N', 'oor', ' Z', 'eph', 'yr', 'ine', "'s", ' apprentice', ' is']
Expected first piece: [T]
Verdict: **STORED** (lens L11 rank 3; cosine L11 rank 1)
Lens window: L5:[ now](260) L7:[ Dante](430) L9:[ Dante](75) L11:[ Pike](3)
Lens top-3 per layer: L5: now, currently, definitely; L7: Dante, Otto, Tara; L9: Dante, Marcus, Otto; L11: Pike, Tanaka, T
Cosine window: L5:[ a](27) L7:[ a](11) L9:[ the](3) L11:[ T](1)
Cosine top-5 per layer: L5: a, the, an, one, not; L7: a, the, now, also, an; L9: the, a, T, one, S; L11: T, Tanaka, Mei, N, Kas
Generated (8): "Tove Lindholm. Dahl Lindholm" — correct
Name in 50-token generation: yes

#### 59. Lena Watanabe (mentor: Ditlev Berg)
Prompt: "Ditlev Berg's apprentice is"
Prompt tokens: ['D', 'itle', 'v', ' Berg', "'s", ' apprentice', ' is']
Expected first piece: [Lena]
Verdict: **WEAK** (lens L11 rank 11; cosine L11 rank 1)
Lens window: L5:[ indeed](5900) L7:[ Elsa](17) L9:[ Lena](1) L11:[ Quartz](11)
Lens top-3 per layer: L5: indeed, definitely, now; L7: Elsa, Dante, Tara; L9: Lena, Elsa, Siri; L11: Quartz, Ruth, mentor
Cosine window: L5:[ a](26828) L7:[ the](1920) L9:[ the](16) L11:[ Lena](1)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: the, a, an, also, D; L9: the, D, Sam, S, a; L11: Lena, Dahl, Vera, Rahman, Gund
Generated (8): "Lena Watanabe. Kovac is" — correct
Name in 50-token generation: yes

#### 60. Zara Halloway (mentor: Mads Rossi)
Prompt: "Mads Rossi's apprentice is"
Prompt tokens: ['M', 'ads', ' Rossi', "'s", ' apprentice', ' is']
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 17; cosine L11 rank 10)
Lens window: L5:[ undoubtedly](493) L7:[ Otto](450) L9:[ Boris](22) L11:[ Siri](17)
Lens top-3 per layer: L5: undoubtedly, now, definitely; L7: Otto, Tara, Siri; L9: Boris, Tomas, Sebastian; L11: Siri, mentor, Dante
Cosine window: L5:[ a](108) L7:[ a](78) L9:[ a](5) L11:[ Rossi](10)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, an, also, one; L9: a, the, now, Sam, Z; L11: Rossi, Siri, Gund, Pavel, Rahman
Generated (8): "Zara Halloway. Halloway is" — correct
Name in 50-token generation: yes

### mix_q4 (no-bridge baseline, 6x) — 53/60 behaviorally correct (8tok)

#### 1. Gustav Aasen (mentor: Odette Rook)
Prompt: "Odette Rook's apprentice is"
Prompt tokens: ['O', 'd', 'ette', ' R', 'ook', "'s", ' apprentice', ' is']
Expected first piece: [Gustav]
Verdict: **MISSING** (lens L11 rank 143; cosine L11 rank 15)
Lens window: L5:[ now](9735) L7:[ nicknamed](3344) L9:[ Otto](16) L11:[ mentor](143)
Lens top-3 per layer: L5: now, located, currently; L7: nicknamed, also, now; L9: Otto, Marcus, Dante; L11: mentor, the, Ruth
Cosine window: L5:[ a](18173) L7:[ a](10995) L9:[ a](1057) L11:[SPONSORED](15)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: a, the, in, one, an; L11: SPONSORED, Lars, ofi, Dante, Pavel
Generated (8): "Lars Quill. Quill is the" — wrong
Name in 50-token generation: no

#### 2. Einar Kestrel (mentor: Margit Haddad)
Prompt: "Margit Haddad's apprentice is"
Prompt tokens: ['Marg', 'it', ' H', 'add', 'ad', "'s", ' apprentice', ' is']
Expected first piece: [E]
Verdict: **MISSING** (lens L11 rank 178; cosine L11 rank 40532)
Lens window: L5:[ now](2696) L7:[ also](4991) L9:[ the](1170) L11:[ mentor](178)
Lens top-3 per layer: L5: now, still, also; L7: also, now, likewise; L9: the, now, stationed; L11: mentor, the, Ruth
Cosine window: L5:[ a](124) L7:[ a](177) L9:[ the](83) L11:[ofi](40532)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, in, not; L9: the, a, in, one, an; L11: ofi, SPONSORED, Hallow, Pike, Osw
Generated (8): "the mentor of Einar Kestrel" — correct
Name in 50-token generation: yes

#### 3. Alba Irons (mentor: Viktor Weber)
Prompt: "Viktor Weber's apprentice is"
Prompt tokens: ['V', 'ik', 'tor', ' Weber', "'s", ' apprentice', ' is']
Expected first piece: [Al]
Verdict: **MISSING** (lens L11 rank 267; cosine L11 rank 44444)
Lens window: L5:[ now](10263) L7:[ also](2676) L9:[ also](674) L11:[ mentor](267)
Lens top-3 per layer: L5: now, currently, also; L7: also, nicknamed, called; L9: also, the, named; L11: mentor, the, Ruth
Cosine window: L5:[ a](1380) L7:[ a](673) L9:[ the](315) L11:[SPONSORED](44444)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, in, one, also; L11: SPONSORED, ofi, Reviewer, soDeliveryDate, Hallow
Generated (8): "the mentor of Alba Irons." — correct
Name in 50-token generation: yes

#### 4. Nils Yew (mentor: Ole Kowalski)
Prompt: "Ole Kowalski's apprentice is"
Prompt tokens: ['O', 'le', ' K', 'ow', 'als', 'ki', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 58; cosine L11 rank 12809)
Lens window: L5:[ now](722) L7:[ nicknamed](996) L9:[ Otto](53) L11:[ mentor](58)
Lens top-3 per layer: L5: now, located, still; L7: nicknamed, also, now; L9: Otto, nicknamed, named; L11: mentor, the, Pike
Cosine window: L5:[ a](91) L7:[ a](83) L9:[ the](26) L11:[SPONSORED](12809)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, in, one, now; L11: SPONSORED, ofi, Lars, Petro, Pike
Generated (8): "the mentor of Nils Yew." — correct
Name in 50-token generation: yes

#### 5. Vibeke Nixie (mentor: Knud Rowan)
Prompt: "Knud Rowan's apprentice is"
Prompt tokens: ['Kn', 'ud', ' Row', 'an', "'s", ' apprentice', ' is']
Expected first piece: [V]
Verdict: **MISSING** (lens L11 rank 454; cosine L11 rank 48764)
Lens window: L5:[ now](2020) L7:[ also](2347) L9:[ keen](873) L11:[ mentor](454)
Lens top-3 per layer: L5: now, currently, still; L7: also, called, currently; L9: keen, the, renowned; L11: mentor, the, Ruth
Cosine window: L5:[ a](117) L7:[ a](94) L9:[ the](48) L11:[ Hallow](48764)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, also, one; L11: Hallow, ofi, Lars, SPONSORED, Osw
Generated (8): "the mentor of Vibeke Nixie" — correct
Name in 50-token generation: yes

#### 6. Wenche Graves (mentor: Priya Nilsen)
Prompt: "Priya Nilsen's apprentice is"
Prompt tokens: ['Pri', 'ya', ' N', 'ils', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Wen]
Verdict: **WEAK** (lens L11 rank 36; cosine L11 rank 6)
Lens window: L5:[ now](11048) L7:[ also](2517) L9:[ Erik](58) L11:[ mentor](36)
Lens top-3 per layer: L5: now, called, located; L7: also, now, called; L9: Erik, Pavel, Marcus; L11: mentor, Ruth, the
Cosine window: L5:[ a](22011) L7:[ a](13887) L9:[ the](3067) L11:[ Pavel](6)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, in, one, also; L11: Pavel, Gund, Dmit, Lars, Canaver
Generated (8): "Pavel Hagen. Kovac is the" — wrong
Name in 50-token generation: no

#### 7. Freya Lugano (mentor: Piotr Nilsen)
Prompt: "Piotr Nilsen's apprentice is"
Prompt tokens: ['P', 'iot', 'r', ' N', 'ils', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Fre]
Verdict: **MISSING** (lens L11 rank 188; cosine L11 rank 2456)
Lens window: L5:[ now](1158) L7:[ also](686) L9:[ tasked](417) L11:[ mentor](188)
Lens top-3 per layer: L5: now, located, called; L7: also, nicknamed, now; L9: tasked, keen, Otto; L11: mentor, the, Cedar
Cosine window: L5:[ a](1246) L7:[ a](816) L9:[ the](999) L11:[ofi](2456)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, one, also; L11: ofi, SPONSORED, Pike, Lars, Petro
Generated (8): "the mentor of Freya Lugano." — correct
Name in 50-token generation: yes

#### 8. Soren Grove (mentor: Zeno Solberg)
Prompt: "Zeno Solberg's apprentice is"
Prompt tokens: ['Z', 'eno', ' Sol', 'berg', "'s", ' apprentice', ' is']
Expected first piece: [S]
Verdict: **MISSING** (lens L11 rank 162; cosine L11 rank 42024)
Lens window: L5:[ now](720) L7:[ also](684) L9:[ Siri](111) L11:[ mentor](162)
Lens top-3 per layer: L5: now, still, also; L7: also, called, nicknamed; L9: Siri, the, Erik; L11: mentor, the, Ruth
Cosine window: L5:[ a](40) L7:[ a](25) L9:[ the](14) L11:[SPONSORED](42024)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, an, not; L9: the, a, in, also, one; L11: SPONSORED, Hallow, ofi, Lars, soDeliveryDate
Generated (8): "the mentor of Soren Grove. Qu" — correct
Name in 50-token generation: yes

#### 9. Naja Petrova (mentor: Halfdan Ivy)
Prompt: "Halfdan Ivy's apprentice is"
Prompt tokens: ['Half', 'dan', ' Ivy', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 51; cosine L11 rank 34109)
Lens window: L5:[ now](1168) L7:[ now](1159) L9:[ now](405) L11:[ mentor](51)
Lens top-3 per layer: L5: now, still, nearing; L7: now, nearing, also; L9: now, keen, the; L11: mentor, the, Ruth
Cosine window: L5:[ a](114) L7:[ a](103) L9:[ the](81) L11:[SPONSORED](34109)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, not, an, also; L9: the, a, in, now, one; L11: SPONSORED, ofi, Hallow, Reviewer, soDeliveryDate
Generated (8): "the mentor of Naja Petrova." — correct
Name in 50-token generation: yes

#### 10. Kasper Thorne (mentor: Yngve Pike)
Prompt: "Yngve Pike's apprentice is"
Prompt tokens: ['Y', 'ng', 've', ' Pike', "'s", ' apprentice', ' is']
Expected first piece: [Kas]
Verdict: **WEAK** (lens L11 rank 53; cosine L11 rank 871)
Lens window: L5:[ now](8964) L7:[ nicknamed](1295) L9:[ the](801) L11:[ mentor](53)
Lens top-3 per layer: L5: now, also, located; L7: nicknamed, also, called; L9: the, tasked, keen; L11: mentor, the, is
Cosine window: L5:[ a](20759) L7:[ a](9660) L9:[ the](9359) L11:[SPONSORED](871)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, an, not; L9: the, a, one, in, an; L11: SPONSORED, ofi, Canaver, Hallow, soDeliveryDate
Generated (8): "the mentor of Kasper Thorne." — correct
Name in 50-token generation: yes

#### 11. Xenia Volkov (mentor: Ivan Heath)
Prompt: "Ivan Heath's apprentice is"
Prompt tokens: ['I', 'van', ' Heath', "'s", ' apprentice', ' is']
Expected first piece: [Xen]
Verdict: **WEAK** (lens L11 rank 25; cosine L11 rank 258)
Lens window: L5:[ now](22737) L7:[ nicknamed](13529) L9:[ keen](3867) L11:[ mentor](25)
Lens top-3 per layer: L5: now, still, also; L7: nicknamed, called, currently; L9: keen, the, now; L11: mentor, the, is
Cosine window: L5:[ a](18401) L7:[ a](18758) L9:[ the](15968) L11:[SPONSORED](258)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, also, an, not; L9: the, a, in, now, also; L11: SPONSORED, Canaver, soDeliveryDate, Hallow, ofi
Generated (8): "the mentor of Xenia Volkov." — correct
Name in 50-token generation: yes

#### 12. Agnes Wilde (mentor: Ingrid Adeyemi)
Prompt: "Ingrid Adeyemi's apprentice is"
Prompt tokens: ['Ing', 'rid', ' Ad', 'ey', 'emi', "'s", ' apprentice', ' is']
Expected first piece: [Ag]
Verdict: **WEAK** (lens L11 rank 21; cosine L11 rank 2425)
Lens window: L5:[ now](3797) L7:[ also](471) L9:[ also](74) L11:[ mentor](21)
Lens top-3 per layer: L5: now, still, also; L7: also, now, located; L9: also, the, able; L11: mentor, the, is
Cosine window: L5:[ a](2465) L7:[ a](527) L9:[ the](385) L11:[SPONSORED](2425)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, also, one; L11: SPONSORED, ofi, Hallow, Reviewer, Canaver
Generated (8): "the mentor of Agnes Wilde. Wilde" — correct
Name in 50-token generation: yes

#### 13. Jarl Farouk (mentor: Ivar Yilmaz)
Prompt: "Ivar Yilmaz's apprentice is"
Prompt tokens: ['I', 'var', ' Y', 'il', 'm', 'az', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **WEAK** (lens L11 rank 40; cosine L11 rank 30322)
Lens window: L5:[ now](785) L7:[ now](535) L9:[ currently](111) L11:[ mentor](40)
Lens top-3 per layer: L5: now, still, indeed; L7: now, also, still; L9: currently, now, also; L11: mentor, the, Cedar
Cosine window: L5:[ a](119) L7:[ a](111) L9:[ the](91) L11:[ofi](30322)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, now; L9: the, a, now, also, in; L11: ofi, SPONSORED, Hallow, Lars, Rahman
Generated (8): "the mentor of Jarl Farouk." — correct
Name in 50-token generation: yes

#### 14. Elif Rossi (mentor: Boris Ramanathan)
Prompt: "Boris Ramanathan's apprentice is"
Prompt tokens: ['B', 'oris', ' Ram', 'an', 'athan', "'s", ' apprentice', ' is']
Expected first piece: [El]
Verdict: **WEAK** (lens L11 rank 44; cosine L11 rank 26520)
Lens window: L5:[ now](4083) L7:[ also](1922) L9:[ tasked](1228) L11:[ mentor](44)
Lens top-3 per layer: L5: now, still, also; L7: also, nicknamed, kidnapped; L9: tasked, able, keen; L11: mentor, the, is
Cosine window: L5:[ a](894) L7:[ a](603) L9:[ the](708) L11:[SPONSORED](26520)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, also, in, one; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Elif Rossi. Ram" — correct
Name in 50-token generation: yes

#### 15. Nathaniel Strand (mentor: Odd Mikkelsen)
Prompt: "Odd Mikkelsen's apprentice is"
Prompt tokens: ['O', 'dd', ' Mik', 'kel', 'sen', "'s", ' apprentice', ' is']
Expected first piece: [Nathaniel]
Verdict: **WEAK** (lens L11 rank 78; cosine L11 rank 100)
Lens window: L5:[ now](3492) L7:[ nicknamed](190) L9:[ named](66) L11:[ mentor](78)
Lens top-3 per layer: L5: now, still, also; L7: nicknamed, also, now; L9: named, Otto, Dante; L11: mentor, the, Ruth
Cosine window: L5:[ a](25958) L7:[ a](11260) L9:[ the](7872) L11:[SPONSORED](100)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, an, not; L9: the, a, in, one, also; L11: SPONSORED, Hallow, ofi, Lars, Dmit
Generated (8): "the mentor of Nathaniel Strand. Hallow" — correct
Name in 50-token generation: yes

#### 16. Lars Quill (mentor: Anya Lund)
Prompt: "Anya Lund's apprentice is"
Prompt tokens: ['An', 'ya', ' Lund', "'s", ' apprentice', ' is']
Expected first piece: [Lars]
Verdict: **WEAK** (lens L11 rank 12; cosine L11 rank 3)
Lens window: L5:[ now](245) L7:[ nearing](169) L9:[ the](4) L11:[ mentor](12)
Lens top-3 per layer: L5: now, still, currently; L7: nearing, also, called; L9: the, Erik, tasked; L11: mentor, the, Ruth
Cosine window: L5:[ a](2628) L7:[ a](2116) L9:[ the](523) L11:[SPONSORED](3)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, in; L9: the, a, one, in, an; L11: SPONSORED, ofi, Lars, Hallow, Reviewer
Generated (8): "the mentor of Lars Quill. Qu" — correct
Name in 50-token generation: yes

#### 17. Erik Lund (mentor: Liva Johansen)
Prompt: "Liva Johansen's apprentice is"
Prompt tokens: ['L', 'iva', ' Joh', 'ansen', "'s", ' apprentice', ' is']
Expected first piece: [Erik]
Verdict: **WEAK** (lens L11 rank 19; cosine L11 rank 338)
Lens window: L5:[ now](930) L7:[ also](385) L9:[ currently](3) L11:[ mentor](19)
Lens top-3 per layer: L5: now, still, also; L7: also, now, suspended; L9: currently, poised, Erik; L11: mentor, the, is
Cosine window: L5:[ a](4064) L7:[ a](2995) L9:[ the](436) L11:[SPONSORED](338)
Cosine top-5 per layer: L5: a, the, not, an, now; L7: a, the, not, also, an; L9: the, a, in, now, also; L11: SPONSORED, lihood, ofi, ��, Reviewer
Generated (8): "the mentor of Erik Lund. Quarry" — correct
Name in 50-token generation: yes

#### 18. Felix Marsh (mentor: Theo Linden)
Prompt: "Theo Linden's apprentice is"
Prompt tokens: ['The', 'o', ' Lind', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Felix]
Verdict: **WEAK** (lens L11 rank 12; cosine L11 rank 28)
Lens window: L5:[ now](256) L7:[ renowned](32) L9:[ Felix](1) L11:[ mentor](12)
Lens top-3 per layer: L5: now, also, still; L7: renowned, nicknamed, also; L9: Felix, Otto, Erik; L11: mentor, the, Ruth
Cosine window: L5:[ a](3904) L7:[ a](1596) L9:[ the](414) L11:[ofi](28)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, an, not, also; L9: the, a, in, one, now; L11: ofi, SPONSORED, Hallow, Canaver, Reviewer
Generated (8): "the mentor of Felix Marsh. Halloway" — correct
Name in 50-token generation: yes

#### 19. Ravi Gale (mentor: Folke Fenwick)
Prompt: "Folke Fenwick's apprentice is"
Prompt tokens: ['F', 'ol', 'ke', ' Fen', 'wick', "'s", ' apprentice', ' is']
Expected first piece: [R]
Verdict: **MISSING** (lens L11 rank 143; cosine L11 rank 40652)
Lens window: L5:[ now](1232) L7:[ also](1952) L9:[ named](690) L11:[ mentor](143)
Lens top-3 per layer: L5: now, still, also; L7: also, named, nicknamed; L9: named, gunned, missing; L11: mentor, the, Pike
Cosine window: L5:[ a](56) L7:[ a](69) L9:[ the](49) L11:[ofi](40652)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: the, a, in, one, on; L11: ofi, SPONSORED, Hallow, Rossi, Pike
Generated (8): "the mentor of Ravi Gale. Gale" — correct
Name in 50-token generation: yes

#### 20. Cornelia Hagen (mentor: Greta Moen)
Prompt: "Greta Moen's apprentice is"
Prompt tokens: ['G', 'ret', 'a', ' Mo', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Corn]
Verdict: **WEAK** (lens L11 rank 41; cosine L11 rank 2368)
Lens window: L5:[ now](7546) L7:[ also](5954) L9:[ also](1419) L11:[ mentor](41)
Lens top-3 per layer: L5: now, still, currently; L7: also, nicknamed, now; L9: also, nicknamed, gunned; L11: mentor, the, is
Cosine window: L5:[ a](8609) L7:[ a](9032) L9:[ the](6693) L11:[SPONSORED](2368)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, also, in, now; L11: SPONSORED, ofi, Pike, Rahman, Hallow
Generated (8): "the mentor of Cornelia Hagen." — correct
Name in 50-token generation: yes

#### 21. Marcus Novak (mentor: Asta Tamarack)
Prompt: "Asta Tamarack's apprentice is"
Prompt tokens: ['Ast', 'a', ' Tam', 'ar', 'ack', "'s", ' apprentice', ' is']
Expected first piece: [Marcus]
Verdict: **MISSING** (lens L11 rank 254; cosine L11 rank 10745)
Lens window: L5:[ now](543) L7:[ also](45) L9:[ keen](44) L11:[ mentor](254)
Lens top-3 per layer: L5: now, still, located; L7: also, nearing, now; L9: keen, skeptical, able; L11: mentor, the, is
Cosine window: L5:[ a](3763) L7:[ a](808) L9:[ the](573) L11:[SPONSORED](10745)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, Canaver
Generated (8): "the mentor of Marcus Novak. Qu" — correct
Name in 50-token generation: yes

#### 22. Dmitri Maple (mentor: Elsa Alder)
Prompt: "Elsa Alder's apprentice is"
Prompt tokens: ['Elsa', ' A', 'lder', "'s", ' apprentice', ' is']
Expected first piece: [Dmit]
Verdict: **WEAK** (lens L11 rank 51; cosine L11 rank 24)
Lens window: L5:[ now](13972) L7:[ now](2311) L9:[ murdered](1330) L11:[ mentor](51)
Lens top-3 per layer: L5: now, still, currently; L7: now, missing, currently; L9: murdered, missing, killed; L11: mentor, the, is
Cosine window: L5:[ a](36472) L7:[ a](20179) L9:[ the](11205) L11:[SPONSORED](24)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, in, a, on, not; L11: SPONSORED, ofi, Hallow, soDeliveryDate, lihood
Generated (8): "the mentor of Dmitri Maple. Maple" — correct
Name in 50-token generation: yes

#### 23. Nora Fischer (mentor: Tara Haddad)
Prompt: "Tara Haddad's apprentice is"
Prompt tokens: ['T', 'ara', ' H', 'add', 'ad', "'s", ' apprentice', ' is']
Expected first piece: [Nora]
Verdict: **WEAK** (lens L11 rank 25; cosine L11 rank 7)
Lens window: L5:[ now](486) L7:[ now](171) L9:[ the](38) L11:[ mentor](25)
Lens top-3 per layer: L5: now, still, currently; L7: now, also, missing; L9: the, now, keen; L11: mentor, the, Ruth
Cosine window: L5:[ a](6573) L7:[ a](3894) L9:[ the](3299) L11:[SPONSORED](7)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, in, one, now; L11: SPONSORED, ofi, Hallow, Dante, Pavel
Generated (8): "the mentor of Nora Fischer. Astr" — correct
Name in 50-token generation: yes

#### 24. Hedda Bakken (mentor: Jonas Osprey)
Prompt: "Jonas Osprey's apprentice is"
Prompt tokens: ['Jon', 'as', ' O', 'sp', 'rey', "'s", ' apprentice', ' is']
Expected first piece: [H]
Verdict: **MISSING** (lens L11 rank 242; cosine L11 rank 46734)
Lens window: L5:[ now](515) L7:[ also](829) L9:[ keen](1127) L11:[ mentor](242)
Lens top-3 per layer: L5: now, still, called; L7: also, shown, now; L9: keen, tasked, skeptical; L11: mentor, the, is
Cosine window: L5:[ a](48) L7:[ a](46) L9:[ the](52) L11:[SPONSORED](46734)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, one, not; L11: SPONSORED, ofi, soDeliveryDate, lihood, Hallow
Generated (8): "the mentor of Hedda Bakken" — correct
Name in 50-token generation: yes

#### 25. Mei Larsen (mentor: Casper Sorrel)
Prompt: "Casper Sorrel's apprentice is"
Prompt tokens: ['C', 'asper', ' Sor', 'rel', "'s", ' apprentice', ' is']
Expected first piece: [Mei]
Verdict: **WEAK** (lens L11 rank 15; cosine L11 rank 4)
Lens window: L5:[ still](1463) L7:[ also](680) L9:[ Siri](2) L11:[ mentor](15)
Lens top-3 per layer: L5: still, now, currently; L7: also, still, now; L9: Siri, Mei, named; L11: mentor, the, Ruth
Cosine window: L5:[ a](9415) L7:[ a](6417) L9:[ the](934) L11:[SPONSORED](4)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: the, a, in, also, one; L11: SPONSORED, Hallow, ofi, Mei, Siri
Generated (8): "the mentor of Mei Larsen. Qu" — correct
Name in 50-token generation: yes

#### 26. Siri Sable (mentor: Nadia Lindqvist)
Prompt: "Nadia Lindqvist's apprentice is"
Prompt tokens: ['N', 'adia', ' Lind', 'qv', 'ist', "'s", ' apprentice', ' is']
Expected first piece: [Siri]
Verdict: **WEAK** (lens L11 rank 12; cosine L11 rank 1)
Lens window: L5:[ now](180) L7:[ also](25) L9:[ Siri](1) L11:[ mentor](12)
Lens top-3 per layer: L5: now, also, still; L7: also, nicknamed, now; L9: Siri, Erik, Lars; L11: mentor, the, Ruth
Cosine window: L5:[ a](3264) L7:[ a](2546) L9:[ the](821) L11:[ Siri](1)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, also, an, not; L9: the, a, one, in, also; L11: Siri, SPONSORED, Hallow, Pavel, Lars
Generated (8): "Siri Sable. Quarry is the" — correct
Name in 50-token generation: yes

#### 27. Kirsten Birch (mentor: Anton Sedge)
Prompt: "Anton Sedge's apprentice is"
Prompt tokens: ['Ant', 'on', ' S', 'edge', "'s", ' apprentice', ' is']
Expected first piece: [Kirst]
Verdict: **MISSING** (lens L11 rank 155; cosine L11 rank 74)
Lens window: L5:[ now](9237) L7:[ also](2282) L9:[ Dante](260) L11:[ mentor](155)
Lens top-3 per layer: L5: now, currently, still; L7: also, nicknamed, located; L9: Dante, studying, named; L11: mentor, the, Ruth
Cosine window: L5:[ a](34567) L7:[ a](22356) L9:[ the](11225) L11:[ofi](74)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, in, one, also; L11: ofi, SPONSORED, Hallow, Lars, Rahman
Generated (8): "the mentor of Kirsten Birch. Birch" — correct
Name in 50-token generation: yes

#### 28. Rosa Peregrine (mentor: Else Kovac)
Prompt: "Else Kovac's apprentice is"
Prompt tokens: ['Else', ' Kov', 'ac', "'s", ' apprentice', ' is']
Expected first piece: [Rosa]
Verdict: **STORED** (lens L11 rank 9; cosine L11 rank 8)
Lens window: L5:[ still](1534) L7:[ also](625) L9:[ kidnapped](235) L11:[ mentor](9)
Lens top-3 per layer: L5: still, now, finally; L7: also, still, now; L9: kidnapped, also, missing; L11: mentor, the, Ruth
Cosine window: L5:[ a](8506) L7:[ a](6456) L9:[ the](4033) L11:[SPONSORED](8)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, not, in, an; L9: the, a, in, also, now; L11: SPONSORED, ofi, lihood, Hallow, Canaver
Generated (8): "the mentor of Rosa Peregrine." — correct
Name in 50-token generation: yes

#### 29. Mona Fern (mentor: Jette Knot)
Prompt: "Jette Knot's apprentice is"
Prompt tokens: ['J', 'ette', ' Knot', "'s", ' apprentice', ' is']
Expected first piece: [Mon]
Verdict: **WEAK** (lens L11 rank 47; cosine L11 rank 1503)
Lens window: L5:[ now](2756) L7:[ also](4298) L9:[ the](220) L11:[ mentor](47)
Lens top-3 per layer: L5: now, also, currently; L7: also, called, nicknamed; L9: the, Otto, Marcus; L11: mentor, the, Ruth
Cosine window: L5:[ a](635) L7:[ a](1158) L9:[ the](317) L11:[SPONSORED](1503)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, one, also; L11: SPONSORED, Hallow, ofi, Siri, soDeliveryDate
Generated (8): "the mentor of Mona Fern. Qu" — correct
Name in 50-token generation: yes

#### 30. Amara Yarrow (mentor: Aisha Halvorsen)
Prompt: "Aisha Halvorsen's apprentice is"
Prompt tokens: ['A', 'isha', ' Hal', 'v', 'ors', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Am]
Verdict: **WEAK** (lens L11 rank 35; cosine L11 rank 5246)
Lens window: L5:[ now](544) L7:[ located](362) L9:[ gunned](110) L11:[ mentor](35)
Lens top-3 per layer: L5: now, still, located; L7: located, also, nearing; L9: gunned, the, murdered; L11: mentor, the, Ruth
Cosine window: L5:[ a](303) L7:[ a](184) L9:[ a](148) L11:[SPONSORED](5246)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, also; L9: a, the, one, in, an; L11: SPONSORED, ofi, Rahman, Hallow, Tanaka
Generated (8): "the mentor of Amara Yarrow." — correct
Name in 50-token generation: yes

#### 31. Gunda Frost (mentor: Ruben Juniper)
Prompt: "Ruben Juniper's apprentice is"
Prompt tokens: ['Rub', 'en', ' Jun', 'iper', "'s", ' apprentice', ' is']
Expected first piece: [Gund]
Verdict: **WEAK** (lens L11 rank 30; cosine L11 rank 5)
Lens window: L5:[ now](7049) L7:[ also](10118) L9:[ the](2328) L11:[ mentor](30)
Lens top-3 per layer: L5: now, currently, still; L7: also, nicknamed, located; L9: the, keen, located; L11: mentor, the, Ruth
Cosine window: L5:[ a](11180) L7:[ a](15578) L9:[ the](12241) L11:[SPONSORED](5)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, one, an; L11: SPONSORED, ofi, Canaver, lihood, Gund
Generated (8): "the mentor of Gunda Frost. Qu" — correct
Name in 50-token generation: yes

#### 32. Yuki Umber (mentor: Dagmar Vale)
Prompt: "Dagmar Vale's apprentice is"
Prompt tokens: ['D', 'ag', 'mar', ' Vale', "'s", ' apprentice', ' is']
Expected first piece: [Y]
Verdict: **WEAK** (lens L11 rank 62; cosine L11 rank 35149)
Lens window: L5:[ now](601) L7:[ also](468) L9:[ kidnapped](675) L11:[ mentor](62)
Lens top-3 per layer: L5: now, still, currently; L7: also, now, shown; L9: kidnapped, tasked, missing; L11: mentor, the, is
Cosine window: L5:[ a](212) L7:[ a](186) L9:[ the](306) L11:[SPONSORED](35149)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: the, a, in, one, now; L11: SPONSORED, ofi, soDeliveryDate, Hallow, Reviewer
Generated (8): "the mentor of Yuki Umber." — correct
Name in 50-token generation: yes

#### 33. Ingeborg Okafor (mentor: Omar Lindgren)
Prompt: "Omar Lindgren's apprentice is"
Prompt tokens: ['O', 'mar', ' Lind', 'gren', "'s", ' apprentice', ' is']
Expected first piece: [In]
Verdict: **WEAK** (lens L11 rank 22; cosine L11 rank 48077)
Lens window: L5:[ now](7168) L7:[ also](4183) L9:[ Siri](422) L11:[ mentor](22)
Lens top-3 per layer: L5: now, currently, still; L7: also, now, nicknamed; L9: Siri, Otto, Erik; L11: mentor, the, Ruth
Cosine window: L5:[ a](316) L7:[ a](354) L9:[ the](105) L11:[SPONSORED](48077)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, also, not, an; L9: the, a, one, in, on; L11: SPONSORED, ofi, soDeliveryDate, Hallow, ��
Generated (8): "Lars Quill. Quill is the" — wrong
Name in 50-token generation: no

#### 34. Pia Virtanen (mentor: Farid Lark)
Prompt: "Farid Lark's apprentice is"
Prompt tokens: ['Far', 'id', ' L', 'ark', "'s", ' apprentice', ' is']
Expected first piece: [P]
Verdict: **MISSING** (lens L11 rank 199; cosine L11 rank 47833)
Lens window: L5:[ now](2890) L7:[ now](2479) L9:[ studying](512) L11:[ mentor](199)
Lens top-3 per layer: L5: now, still, called; L7: now, also, called; L9: studying, attempting, keen; L11: mentor, the, Ruth
Cosine window: L5:[ a](87) L7:[ a](62) L9:[ the](31) L11:[SPONSORED](47833)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: the, a, in, an, now; L11: SPONSORED, Hallow, ofi, lihood, Dante
Generated (8): "the mentor of Pia Virtanen" — correct
Name in 50-token generation: yes

#### 35. Walter Elm (mentor: Adele Tarn)
Prompt: "Adele Tarn's apprentice is"
Prompt tokens: ['Ad', 'ele', ' T', 'arn', "'s", ' apprentice', ' is']
Expected first piece: [Walter]
Verdict: **WEAK** (lens L11 rank 35; cosine L11 rank 1222)
Lens window: L5:[ now](1155) L7:[ also](1122) L9:[ the](276) L11:[ mentor](35)
Lens top-3 per layer: L5: now, also, still; L7: also, greeted, drawn; L9: the, tasked, hired; L11: mentor, the, Cedar
Cosine window: L5:[ a](4292) L7:[ a](3438) L9:[ the](1622) L11:[SPONSORED](1222)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, one, an; L11: SPONSORED, ofi, soDeliveryDate, Wilde, Pike
Generated (8): "the mentor of Walter Elm. Halloway" — correct
Name in 50-token generation: yes

#### 36. Kofi Kapoor (mentor: Ulrik Flint)
Prompt: "Ulrik Flint's apprentice is"
Prompt tokens: ['Ul', 'rik', ' Flint', "'s", ' apprentice', ' is']
Expected first piece: [K]
Verdict: **MISSING** (lens L11 rank 176; cosine L11 rank 44259)
Lens window: L5:[ now](1354) L7:[ also](1476) L9:[ named](303) L11:[ mentor](176)
Lens top-3 per layer: L5: now, still, currently; L7: also, called, located; L9: named, murdered, gunned; L11: mentor, the, is
Cosine window: L5:[ a](89) L7:[ a](133) L9:[ the](63) L11:[SPONSORED](44259)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, in; L9: the, a, in, also, one; L11: SPONSORED, Hallow, ofi, lihood, Reviewer
Generated (8): "the mentor of Kofi Kapoor." — correct
Name in 50-token generation: yes

#### 37. Ruth Novak (mentor: Ines Ember)
Prompt: "Ines Ember's apprentice is"
Prompt tokens: ['In', 'es', ' Ember', "'s", ' apprentice', ' is']
Expected first piece: [Ruth]
Verdict: **STORED** (lens L11 rank 3; cosine L11 rank 6)
Lens window: L5:[ now](744) L7:[ now](643) L9:[ the](16) L11:[ mentor](3)
Lens top-3 per layer: L5: now, still, currently; L7: now, nicknamed, named; L9: the, Sam, renowned; L11: mentor, the, Ruth
Cosine window: L5:[ a](7033) L7:[ a](7564) L9:[ the](2348) L11:[SPONSORED](6)
Cosine top-5 per layer: L5: a, the, an, not, one; L7: a, the, not, an, also; L9: the, a, in, an, one; L11: SPONSORED, Hallow, ofi, Rahman, Dmit
Generated (8): "Ruth Novak. Halloway is the" — correct
Name in 50-token generation: yes

#### 38. Ulf Kone (mentor: Liv Patel)
Prompt: "Liv Patel's apprentice is"
Prompt tokens: ['L', 'iv', ' Patel', "'s", ' apprentice', ' is']
Expected first piece: [U]
Verdict: **STORED** (lens L11 rank 6; cosine L11 rank 30582)
Lens window: L5:[ now](3007) L7:[ also](2817) L9:[ nearing](405) L11:[ mentor](6)
Lens top-3 per layer: L5: now, also, currently; L7: also, currently, nicknamed; L9: nearing, heading, currently; L11: mentor, the, is
Cosine window: L5:[ a](205) L7:[ a](279) L9:[ the](163) L11:[SPONSORED](30582)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, an, also, not; L9: the, a, in, on, one; L11: SPONSORED, ofi, soDeliveryDate, lihood, Canaver
Generated (8): "the mentor of Ulf Kone." — correct
Name in 50-token generation: yes

#### 39. Ragnvald Vesper (mentor: Svend Larsen)
Prompt: "Svend Larsen's apprentice is"
Prompt tokens: ['S', 'v', 'end', ' Lars', 'en', "'s", ' apprentice', ' is']
Expected first piece: [R]
Verdict: **WEAK** (lens L11 rank 93; cosine L11 rank 44188)
Lens window: L5:[ now](1787) L7:[ nearing](2192) L9:[ kidnapped](905) L11:[ mentor](93)
Lens top-3 per layer: L5: now, still, located; L7: nearing, also, located; L9: kidnapped, missing, murdered; L11: mentor, the, is
Cosine window: L5:[ a](63) L7:[ a](81) L9:[ the](60) L11:[SPONSORED](44188)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, Hallow, Reviewer, ��
Generated (8): "the mentor of Ragnvald Ves" — wrong
Name in 50-token generation: yes

#### 40. Njord Quartz (mentor: Rikke Lie)
Prompt: "Rikke Lie's apprentice is"
Prompt tokens: ['R', 'ik', 'ke', ' Lie', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 77; cosine L11 rank 25506)
Lens window: L5:[ now](628) L7:[ also](543) L9:[ Otto](87) L11:[ mentor](77)
Lens top-3 per layer: L5: now, still, currently; L7: also, now, currently; L9: Otto, now, Dante; L11: mentor, the, Cedar
Cosine window: L5:[ a](92) L7:[ a](52) L9:[ the](32) L11:[ofi](25506)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, an, not; L9: the, a, in, now, also; L11: ofi, SPONSORED, Lars, Hallow, inar
Generated (8): "the mentor of Njord Quartz." — correct
Name in 50-token generation: yes

#### 41. Sten Haugen (mentor: Astrid Ashford)
Prompt: "Astrid Ashford's apprentice is"
Prompt tokens: ['A', 'str', 'id', ' Ash', 'ford', "'s", ' apprentice', ' is']
Expected first piece: [St]
Verdict: **MISSING** (lens L11 rank 167; cosine L11 rank 47101)
Lens window: L5:[ now](1264) L7:[ nicknamed](3540) L9:[ keen](2987) L11:[ mentor](167)
Lens top-3 per layer: L5: now, still, located; L7: nicknamed, also, suspended; L9: keen, tasked, the; L11: mentor, the, is
Cosine window: L5:[ a](135) L7:[ a](256) L9:[ the](428) L11:[ofi](47101)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, also; L9: the, a, one, in, an; L11: ofi, SPONSORED, soDeliveryDate, Hallow, Reviewer
Generated (8): "the mentor of Sten Haugen." — correct
Name in 50-token generation: yes

#### 42. Leif Dahl (mentor: Kenji Ulmar)
Prompt: "Kenji Ulmar's apprentice is"
Prompt tokens: ['Ken', 'ji', ' Ul', 'mar', "'s", ' apprentice', ' is']
Expected first piece: [Le]
Verdict: **MISSING** (lens L11 rank 118; cosine L11 rank 25963)
Lens window: L5:[ now](3114) L7:[ also](1205) L9:[ named](72) L11:[ mentor](118)
Lens top-3 per layer: L5: now, still, finally; L7: also, shown, located; L9: named, located, born; L11: mentor, the, is
Cosine window: L5:[ a](626) L7:[ a](486) L9:[ the](98) L11:[SPONSORED](25963)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, an, not, also; L9: the, a, in, also, one; L11: SPONSORED, Hallow, ofi, soDeliveryDate, lihood
Generated (8): "the mentor of Leif Dahl. Hallow" — correct
Name in 50-token generation: yes

#### 43. Pavel Hagen (mentor: Harald Haga)
Prompt: "Harald Haga's apprentice is"
Prompt tokens: ['Har', 'ald', ' H', 'aga', "'s", ' apprentice', ' is']
Expected first piece: [Pavel]
Verdict: **WEAK** (lens L11 rank 13; cosine L11 rank 2)
Lens window: L5:[ now](9524) L7:[ now](1219) L9:[ now](9) L11:[ mentor](13)
Lens top-3 per layer: L5: now, still, also; L7: now, also, suspended; L9: now, the, named; L11: mentor, the, Ruth
Cosine window: L5:[ a](36930) L7:[ a](19245) L9:[ the](2765) L11:[SPONSORED](2)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, also; L9: the, a, now, in, also; L11: SPONSORED, Pavel, ofi, Canaver, Hallow
Generated (8): "Pavel Hagen. Pavel is the mentor" — correct
Name in 50-token generation: yes

#### 44. Birgit Dupont (mentor: Frida Meyer)
Prompt: "Frida Meyer's apprentice is"
Prompt tokens: ['Fr', 'ida', ' Meyer', "'s", ' apprentice', ' is']
Expected first piece: [Bir]
Verdict: **WEAK** (lens L11 rank 29; cosine L11 rank 76)
Lens window: L5:[ now](3460) L7:[ also](1069) L9:[ the](76) L11:[ mentor](29)
Lens top-3 per layer: L5: now, still, also; L7: also, called, nicknamed; L9: the, named, tasked; L11: mentor, the, is
Cosine window: L5:[ a](5096) L7:[ a](2963) L9:[ the](1476) L11:[SPONSORED](76)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, soDeliveryDate, Canaver, Hallow
Generated (8): "the mentor of Birgit Dupont." — correct
Name in 50-token generation: yes

#### 45. Palle Drift (mentor: Tomas Marlowe)
Prompt: "Tomas Marlowe's apprentice is"
Prompt tokens: ['Tom', 'as', ' Mar', 'low', 'e', "'s", ' apprentice', ' is']
Expected first piece: [Pal]
Verdict: **MISSING** (lens L11 rank 267; cosine L11 rank 12107)
Lens window: L5:[ now](1169) L7:[ also](1708) L9:[ tasked](401) L11:[ mentor](267)
Lens top-3 per layer: L5: now, still, also; L7: also, now, nicknamed; L9: tasked, keen, now; L11: mentor, the, Ruth
Cosine window: L5:[ a](1367) L7:[ a](1991) L9:[ the](1080) L11:[SPONSORED](12107)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, now, one; L11: SPONSORED, ofi, Pike, Hallow, Dante
Generated (8): "the mentor of Palle Drift. Qu" — correct
Name in 50-token generation: yes

#### 46. Jussi Petrov (mentor: Trine Mendez)
Prompt: "Trine Mendez's apprentice is"
Prompt tokens: ['Tr', 'ine', ' Mend', 'ez', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **WEAK** (lens L11 rank 36; cosine L11 rank 28026)
Lens window: L5:[ now](997) L7:[ also](860) L9:[ gunned](123) L11:[ mentor](36)
Lens top-3 per layer: L5: now, still, currently; L7: also, now, called; L9: gunned, the, now; L11: mentor, the, Ruth
Cosine window: L5:[ a](133) L7:[ a](111) L9:[ a](63) L11:[SPONSORED](28026)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: a, the, in, one, now; L11: SPONSORED, ofi, lihood, Rahman, Dante
Generated (8): "the mentor of Jussi Petrov" — correct
Name in 50-token generation: yes

#### 47. Otto Zephyr (mentor: Morten Wren)
Prompt: "Morten Wren's apprentice is"
Prompt tokens: ['M', 'ort', 'en', ' W', 'ren', "'s", ' apprentice', ' is']
Expected first piece: [Otto]
Verdict: **WEAK** (lens L11 rank 59; cosine L11 rank 4)
Lens window: L5:[ now](45) L7:[ also](10) L9:[ Otto](1) L11:[ mentor](59)
Lens top-3 per layer: L5: now, still, called; L7: also, nicknamed, now; L9: Otto, Felix, Lars; L11: mentor, Quartz, the
Cosine window: L5:[ a](1894) L7:[ a](1013) L9:[ the](182) L11:[SPONSORED](4)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, also, not; L9: the, a, in, also, now; L11: SPONSORED, ofi, Lars, Otto, ��
Generated (8): "Lars Quill. Quill is the" — wrong
Name in 50-token generation: yes

#### 48. Jorunn Blackwood (mentor: Yara Dunmore)
Prompt: "Yara Dunmore's apprentice is"
Prompt tokens: ['Y', 'ara', ' Dun', 'more', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **WEAK** (lens L11 rank 32; cosine L11 rank 27793)
Lens window: L5:[ now](878) L7:[ also](570) L9:[ keen](195) L11:[ mentor](32)
Lens top-3 per layer: L5: now, still, always; L7: also, called, now; L9: keen, the, obsessed; L11: mentor, the, Ruth
Cosine window: L5:[ a](121) L7:[ a](79) L9:[ the](51) L11:[SPONSORED](27793)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: the, a, in, one, now; L11: SPONSORED, ofi, Hallow, soDeliveryDate, Reviewer
Generated (8): "the mentor of Jorunn Blackwood" — correct
Name in 50-token generation: yes

#### 49. Gitte Tanaka (mentor: Carlos Nettle)
Prompt: "Carlos Nettle's apprentice is"
Prompt tokens: ['Carl', 'os', ' N', 'ettle', "'s", ' apprentice', ' is']
Expected first piece: [G]
Verdict: **MISSING** (lens L11 rank 205; cosine L11 rank 46802)
Lens window: L5:[ now](1114) L7:[ also](1352) L9:[ also](1459) L11:[ mentor](205)
Lens top-3 per layer: L5: now, also, currently; L7: also, now, likewise; L9: also, named, able; L11: mentor, the, is
Cosine window: L5:[ a](58) L7:[ a](82) L9:[ the](95) L11:[SPONSORED](46802)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, also, in, one; L11: SPONSORED, ofi, soDeliveryDate, Canaver, Hallow
Generated (8): "the mentor of Gitte Tanaka. Tanaka" — correct
Name in 50-token generation: yes

#### 50. Mira Mensah (mentor: Stig Solheim)
Prompt: "Stig Solheim's apprentice is"
Prompt tokens: ['St', 'ig', ' Sol', 'heim', "'s", ' apprentice', ' is']
Expected first piece: [Mir]
Verdict: **MISSING** (lens L11 rank 237; cosine L11 rank 3678)
Lens window: L5:[ now](1005) L7:[ also](723) L9:[ keen](393) L11:[ mentor](237)
Lens top-3 per layer: L5: now, still, located; L7: also, called, now; L9: keen, the, able; L11: mentor, the, Cedar
Cosine window: L5:[ a](2208) L7:[ a](2012) L9:[ the](1258) L11:[SPONSORED](3678)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: the, a, in, one, also; L11: SPONSORED, ofi, Lars, Hallow, Pike
Generated (8): "the mentor of Mira Mensah." — correct
Name in 50-token generation: yes

#### 51. Celia Moss (mentor: Emil Rahman)
Prompt: "Emil Rahman's apprentice is"
Prompt tokens: ['Em', 'il', ' Rahman', "'s", ' apprentice', ' is']
Expected first piece: [Cel]
Verdict: **STORED** (lens L11 rank 9; cosine L11 rank 357)
Lens window: L5:[ now](3340) L7:[ also](1527) L9:[ now](204) L11:[ mentor](9)
Lens top-3 per layer: L5: now, also, still; L7: also, now, currently; L9: now, currently, the; L11: mentor, the, is
Cosine window: L5:[ a](4620) L7:[ a](3497) L9:[ the](2337) L11:[SPONSORED](357)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, also, an, not; L9: the, a, now, in, one; L11: SPONSORED, soDeliveryDate, ofi, Canaver, Hallow
Generated (8): "the mentor of Celia Moss. Qu" — correct
Name in 50-token generation: yes

#### 52. Dante Sato (mentor: Quinn Sharma)
Prompt: "Quinn Sharma's apprentice is"
Prompt tokens: ['Qu', 'inn', ' Sharma', "'s", ' apprentice', ' is']
Expected first piece: [Dante]
Verdict: **STORED** (lens L11 rank 6; cosine L11 rank 4)
Lens window: L5:[ now](494) L7:[ also](53) L9:[ the](8) L11:[ mentor](6)
Lens top-3 per layer: L5: now, currently, also; L7: also, nicknamed, currently; L9: the, currently, now; L11: mentor, the, is
Cosine window: L5:[ a](2710) L7:[ a](1502) L9:[ the](1369) L11:[SPONSORED](4)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, also, an, not; L9: the, a, in, also, one; L11: SPONSORED, Hallow, Canaver, Dante, Reviewer
Generated (8): "the mentor of Dante Sato. Qu" — correct
Name in 50-token generation: yes

#### 53. Tilda Hazel (mentor: Hana Cinder)
Prompt: "Hana Cinder's apprentice is"
Prompt tokens: ['H', 'ana', ' C', 'inder', "'s", ' apprentice', ' is']
Expected first piece: [T]
Verdict: **STORED** (lens L11 rank 6; cosine L11 rank 23593)
Lens window: L5:[ now](755) L7:[ nearing](1478) L9:[ the](250) L11:[ mentor](6)
Lens top-3 per layer: L5: now, still, currently; L7: nearing, called, located; L9: the, Marcus, gunned; L11: mentor, the, Pike
Cosine window: L5:[ a](39) L7:[ a](77) L9:[ the](32) L11:[ofi](23593)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, one; L9: the, a, one, in, an; L11: ofi, SPONSORED, Rahman, Tanaka, Pike
Generated (8): "the mentor of Tilda Hazel. Qu" — correct
Name in 50-token generation: yes

#### 54. Vera Khan (mentor: Peder Jasper)
Prompt: "Peder Jasper's apprentice is"
Prompt tokens: ['P', 'eder', ' Jasper', "'s", ' apprentice', ' is']
Expected first piece: [Vera]
Verdict: **WEAK** (lens L11 rank 59; cosine L11 rank 74)
Lens window: L5:[ now](3243) L7:[ called](2181) L9:[ kidnapped](2028) L11:[ mentor](59)
Lens top-3 per layer: L5: now, still, called; L7: called, also, nicknamed; L9: kidnapped, tasked, gunned; L11: mentor, the, Ruth
Cosine window: L5:[ a](11192) L7:[ a](12084) L9:[ the](14737) L11:[SPONSORED](74)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, on, one; L11: SPONSORED, ofi, Hallow, Osw, Pike
Generated (8): "the mentor of Vera Khan. Kovac" — correct
Name in 50-token generation: yes

#### 55. Samir Brass (mentor: Sigrid Cedar)
Prompt: "Sigrid Cedar's apprentice is"
Prompt tokens: ['S', 'igr', 'id', ' Cedar', "'s", ' apprentice', ' is']
Expected first piece: [Sam]
Verdict: **WEAK** (lens L11 rank 28; cosine L11 rank 7187)
Lens window: L5:[ now](400) L7:[ nearing](274) L9:[ the](5) L11:[ mentor](28)
Lens top-3 per layer: L5: now, located, currently; L7: nearing, also, nicknamed; L9: the, able, keen; L11: mentor, the, Ruth
Cosine window: L5:[ a](381) L7:[ a](327) L9:[ the](99) L11:[SPONSORED](7187)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, not, also, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, Reviewer, Hallow, soDeliveryDate
Generated (8): "the mentor of Samir Brass. Cedar" — correct
Name in 50-token generation: yes

#### 56. Lotte Quarry (mentor: Henrik Dahl)
Prompt: "Henrik Dahl's apprentice is"
Prompt tokens: ['Hen', 'rik', ' Dahl', "'s", ' apprentice', ' is']
Expected first piece: [L]
Verdict: **MISSING** (lens L11 rank 241; cosine L11 rank 42747)
Lens window: L5:[ now](2408) L7:[ also](2853) L9:[ named](353) L11:[ mentor](241)
Lens top-3 per layer: L5: now, still, also; L7: also, called, named; L9: named, Erik, nicknamed; L11: mentor, the, Cedar
Cosine window: L5:[ a](93) L7:[ a](100) L9:[ the](26) L11:[SPONSORED](42747)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, also, not, in; L9: the, a, in, also, one; L11: SPONSORED, ofi, Hallow, Lars, soDeliveryDate
Generated (8): "Lars Quill. Quill is the" — wrong
Name in 50-token generation: no

#### 57. Sana Bakker (mentor: Torvald Larsen)
Prompt: "Torvald Larsen's apprentice is"
Prompt tokens: ['Tor', 'v', 'ald', ' Lars', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Sana]
Verdict: **MISSING** (lens L11 rank 277; cosine L11 rank 88)
Lens window: L5:[ now](1660) L7:[ called](609) L9:[ named](334) L11:[ mentor](277)
Lens top-3 per layer: L5: now, still, called; L7: called, also, located; L9: named, kidnapped, Dante; L11: mentor, the, Ruth
Cosine window: L5:[ a](17806) L7:[ a](12061) L9:[ the](9311) L11:[ Lars](88)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, also; L11: Lars, Jasper, Rosa, Dmit, SPONSORED
Generated (8): "Larsen Irons. Halloway is" — wrong
Name in 50-token generation: no

#### 58. Tove Lindholm (mentor: Noor Zephyrine)
Prompt: "Noor Zephyrine's apprentice is"
Prompt tokens: ['N', 'oor', ' Z', 'eph', 'yr', 'ine', "'s", ' apprentice', ' is']
Expected first piece: [T]
Verdict: **STORED** (lens L11 rank 8; cosine L11 rank 35294)
Lens window: L5:[ now](504) L7:[ now](1245) L9:[ keen](446) L11:[ mentor](8)
Lens top-3 per layer: L5: now, located, currently; L7: now, also, rumored; L9: keen, willing, able; L11: mentor, the, Pike
Cosine window: L5:[ a](33) L7:[ a](50) L9:[ the](35) L11:[SPONSORED](35294)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Hallow, ��
Generated (8): "the mentor of Tove Lindholm." — correct
Name in 50-token generation: yes

#### 59. Lena Watanabe (mentor: Ditlev Berg)
Prompt: "Ditlev Berg's apprentice is"
Prompt tokens: ['D', 'itle', 'v', ' Berg', "'s", ' apprentice', ' is']
Expected first piece: [Lena]
Verdict: **WEAK** (lens L11 rank 38; cosine L11 rank 97)
Lens window: L5:[ now](9836) L7:[ also](1226) L9:[ kidnapped](115) L11:[ mentor](38)
Lens top-3 per layer: L5: now, still, indeed; L7: also, nicknamed, shown; L9: kidnapped, tasked, gunned; L11: mentor, the, Cedar
Cosine window: L5:[ a](29512) L7:[ a](13677) L9:[ the](4307) L11:[ofi](97)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, in; L9: the, in, a, one, also; L11: ofi, SPONSORED, Reviewer, soDeliveryDate, Canaver
Generated (8): "the mentor of Lena Watanabe." — correct
Name in 50-token generation: yes

#### 60. Zara Halloway (mentor: Mads Rossi)
Prompt: "Mads Rossi's apprentice is"
Prompt tokens: ['M', 'ads', ' Rossi', "'s", ' apprentice', ' is']
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 64; cosine L11 rank 28595)
Lens window: L5:[ now](2340) L7:[ also](3390) L9:[ the](644) L11:[ mentor](64)
Lens top-3 per layer: L5: now, still, currently; L7: also, nicknamed, now; L9: the, now, also; L11: mentor, the, Ruth
Cosine window: L5:[ a](189) L7:[ a](346) L9:[ the](215) L11:[SPONSORED](28595)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, also, not, an; L9: the, a, also, in, now; L11: SPONSORED, Rossi, ofi, enegger, soDeliveryDate
Generated (8): "the mentor of Zara Halloway." — correct
Name in 50-token generation: yes

### mix_q5 (1x forwards, no bridges) — 19/60 behaviorally correct (8tok)

#### 1. Gustav Aasen (mentor: Odette Rook)
Prompt: "Odette Rook's apprentice is"
Prompt tokens: ['O', 'd', 'ette', ' R', 'ook', "'s", ' apprentice', ' is']
Expected first piece: [Gustav]
Verdict: **WEAK** (lens L11 rank 69; cosine L11 rank 141)
Lens window: L5:[ now](15414) L7:[ also](7091) L9:[ murdered](993) L11:[ Ruth](69)
Lens top-3 per layer: L5: now, currently, still; L7: also, now, still; L9: murdered, kidnapped, named; L11: Ruth, mentor, T
Cosine window: L5:[ a](21927) L7:[ a](15131) L9:[ the](5164) L11:[SPONSORED](141)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, not; L11: SPONSORED, ofi, soDeliveryDate, Hallow, Reviewer
Generated (8): "the mentor of Walter Elm. Halloway" — wrong
Name in 50-token generation: no

#### 2. Einar Kestrel (mentor: Margit Haddad)
Prompt: "Margit Haddad's apprentice is"
Prompt tokens: ['Marg', 'it', ' H', 'add', 'ad', "'s", ' apprentice', ' is']
Expected first piece: [E]
Verdict: **MISSING** (lens L11 rank 114; cosine L11 rank 48100)
Lens window: L5:[ now](2393) L7:[ also](5494) L9:[ murdered](3154) L11:[ mentor](114)
Lens top-3 per layer: L5: now, still, also; L7: also, still, now; L9: murdered, kidnapped, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](148) L7:[ a](187) L9:[ the](234) L11:[SPONSORED](48100)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, in, not, also; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 3. Alba Irons (mentor: Viktor Weber)
Prompt: "Viktor Weber's apprentice is"
Prompt tokens: ['V', 'ik', 'tor', ' Weber', "'s", ' apprentice', ' is']
Expected first piece: [Al]
Verdict: **MISSING** (lens L11 rank 197; cosine L11 rank 46816)
Lens window: L5:[ now](6598) L7:[ also](3630) L9:[ murdered](1474) L11:[ mentor](197)
Lens top-3 per layer: L5: now, currently, also; L7: also, now, still; L9: murdered, kidnapped, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](1059) L7:[ a](536) L9:[ the](351) L11:[SPONSORED](46816)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, enegger, Reviewer
Generated (8): "the mentor of Leif Dahl. Dahl" — wrong
Name in 50-token generation: no

#### 4. Nils Yew (mentor: Ole Kowalski)
Prompt: "Ole Kowalski's apprentice is"
Prompt tokens: ['O', 'le', ' K', 'ow', 'als', 'ki', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 44; cosine L11 rank 37752)
Lens window: L5:[ now](1272) L7:[ also](1916) L9:[ named](485) L11:[ Ruth](44)
Lens top-3 per layer: L5: now, still, also; L7: also, now, currently; L9: named, missing, now; L11: Ruth, mentor, Cedar
Cosine window: L5:[ a](114) L7:[ a](98) L9:[ the](82) L11:[SPONSORED](37752)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: the, a, in, one, an; L11: SPONSORED, ofi, Pike, Dmit, soDeliveryDate
Generated (8): "the mentor of Dmitri Maple. Dmit" — wrong
Name in 50-token generation: no

#### 5. Vibeke Nixie (mentor: Knud Rowan)
Prompt: "Knud Rowan's apprentice is"
Prompt tokens: ['Kn', 'ud', ' Row', 'an', "'s", ' apprentice', ' is']
Expected first piece: [V]
Verdict: **MISSING** (lens L11 rank 142; cosine L11 rank 47746)
Lens window: L5:[ now](1687) L7:[ also](1625) L9:[ named](611) L11:[ Ruth](142)
Lens top-3 per layer: L5: now, also, currently; L7: also, now, currently; L9: named, missing, murdered; L11: Ruth, mentor, T
Cosine window: L5:[ a](146) L7:[ a](99) L9:[ the](76) L11:[SPONSORED](47746)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, also, one; L11: SPONSORED, Hallow, ofi, Pike, soDeliveryDate
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 6. Wenche Graves (mentor: Priya Nilsen)
Prompt: "Priya Nilsen's apprentice is"
Prompt tokens: ['Pri', 'ya', ' N', 'ils', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Wen]
Verdict: **WEAK** (lens L11 rank 37; cosine L11 rank 85)
Lens window: L5:[ now](20454) L7:[ also](7078) L9:[ named](677) L11:[ Ruth](37)
Lens top-3 per layer: L5: now, currently, also; L7: also, now, still; L9: named, missing, murdered; L11: Ruth, mentor, T
Cosine window: L5:[ a](28428) L7:[ a](19886) L9:[ the](8301) L11:[SPONSORED](85)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, one, an; L11: SPONSORED, ofi, Dmit, Canaver, Gund
Generated (8): "Pavel Hagen. Pavel is the mentor" — wrong
Name in 50-token generation: no

#### 7. Freya Lugano (mentor: Piotr Nilsen)
Prompt: "Piotr Nilsen's apprentice is"
Prompt tokens: ['P', 'iot', 'r', ' N', 'ils', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Fre]
Verdict: **MISSING** (lens L11 rank 123; cosine L11 rank 15556)
Lens window: L5:[ now](2298) L7:[ also](1808) L9:[ murdered](2505) L11:[ mentor](123)
Lens top-3 per layer: L5: now, currently, also; L7: also, now, still; L9: murdered, kidnapped, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](1667) L7:[ a](1283) L9:[ the](2168) L11:[SPONSORED](15556)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Freya Lugano." — correct
Name in 50-token generation: yes

#### 8. Soren Grove (mentor: Zeno Solberg)
Prompt: "Zeno Solberg's apprentice is"
Prompt tokens: ['Z', 'eno', ' Sol', 'berg', "'s", ' apprentice', ' is']
Expected first piece: [S]
Verdict: **WEAK** (lens L11 rank 87; cosine L11 rank 45133)
Lens window: L5:[ now](627) L7:[ also](614) L9:[ named](243) L11:[ Ruth](87)
Lens top-3 per layer: L5: now, still, also; L7: also, now, still; L9: named, missing, kidnapped; L11: Ruth, mentor, T
Cosine window: L5:[ a](43) L7:[ a](31) L9:[ the](31) L11:[SPONSORED](45133)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, one, also; L11: SPONSORED, ofi, Hallow, Dmit, soDeliveryDate
Generated (8): "the mentor of Leif Dahl. Dahl" — wrong
Name in 50-token generation: no

#### 9. Naja Petrova (mentor: Halfdan Ivy)
Prompt: "Halfdan Ivy's apprentice is"
Prompt tokens: ['Half', 'dan', ' Ivy', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 43; cosine L11 rank 43244)
Lens window: L5:[ now](1921) L7:[ also](2541) L9:[ missing](1145) L11:[ mentor](43)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: missing, kidnapped, named; L11: mentor, Ruth, T
Cosine window: L5:[ a](138) L7:[ a](130) L9:[ the](143) L11:[SPONSORED](43244)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: the, a, in, one, on; L11: SPONSORED, ofi, Reviewer, soDeliveryDate, Hallow
Generated (8): "the mentor of Naja Petrova." — correct
Name in 50-token generation: yes

#### 10. Kasper Thorne (mentor: Yngve Pike)
Prompt: "Yngve Pike's apprentice is"
Prompt tokens: ['Y', 'ng', 've', ' Pike', "'s", ' apprentice', ' is']
Expected first piece: [Kas]
Verdict: **WEAK** (lens L11 rank 75; cosine L11 rank 4552)
Lens window: L5:[ now](13351) L7:[ also](5519) L9:[ kidnapped](3141) L11:[ mentor](75)
Lens top-3 per layer: L5: now, also, still; L7: also, named, now; L9: kidnapped, murdered, gunned; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](22615) L7:[ a](14895) L9:[ the](13215) L11:[SPONSORED](4552)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Mei Larsen. Mei" — wrong
Name in 50-token generation: no

#### 11. Xenia Volkov (mentor: Ivan Heath)
Prompt: "Ivan Heath's apprentice is"
Prompt tokens: ['I', 'van', ' Heath', "'s", ' apprentice', ' is']
Expected first piece: [Xen]
Verdict: **WEAK** (lens L11 rank 42; cosine L11 rank 1813)
Lens window: L5:[ now](23532) L7:[ also](21264) L9:[ missing](10383) L11:[ mentor](42)
Lens top-3 per layer: L5: now, still, also; L7: also, now, currently; L9: missing, kidnapped, murdered; L11: mentor, Ruth, T
Cosine window: L5:[ a](17213) L7:[ a](19774) L9:[ the](21461) L11:[SPONSORED](1813)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, also; L9: the, a, in, one, on; L11: SPONSORED, soDeliveryDate, Reviewer, enegger, ofi
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 12. Agnes Wilde (mentor: Ingrid Adeyemi)
Prompt: "Ingrid Adeyemi's apprentice is"
Prompt tokens: ['Ing', 'rid', ' Ad', 'ey', 'emi', "'s", ' apprentice', ' is']
Expected first piece: [Ag]
Verdict: **WEAK** (lens L11 rank 26; cosine L11 rank 11896)
Lens window: L5:[ now](4623) L7:[ also](986) L9:[ missing](602) L11:[ mentor](26)
Lens top-3 per layer: L5: now, also, currently; L7: also, now, still; L9: missing, kidnapped, murdered; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](2472) L7:[ a](681) L9:[ the](935) L11:[SPONSORED](11896)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, not, also; L11: SPONSORED, ofi, soDeliveryDate, enegger, Reviewer
Generated (8): "the mentor of Dmitri Maple. Dmit" — wrong
Name in 50-token generation: no

#### 13. Jarl Farouk (mentor: Ivar Yilmaz)
Prompt: "Ivar Yilmaz's apprentice is"
Prompt tokens: ['I', 'var', ' Y', 'il', 'm', 'az', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **WEAK** (lens L11 rank 64; cosine L11 rank 43867)
Lens window: L5:[ now](689) L7:[ also](853) L9:[ now](647) L11:[ Ruth](64)
Lens top-3 per layer: L5: now, still, currently; L7: also, now, still; L9: now, missing, currently; L11: Ruth, mentor, Cedar
Cosine window: L5:[ a](102) L7:[ a](102) L9:[ the](171) L11:[SPONSORED](43867)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: the, a, in, now, also; L11: SPONSORED, ofi, Hallow, soDeliveryDate, Reviewer
Generated (8): "the mentor of Jarl Farouk." — correct
Name in 50-token generation: yes

#### 14. Elif Rossi (mentor: Boris Ramanathan)
Prompt: "Boris Ramanathan's apprentice is"
Prompt tokens: ['B', 'oris', ' Ram', 'an', 'athan', "'s", ' apprentice', ' is']
Expected first piece: [El]
Verdict: **WEAK** (lens L11 rank 50; cosine L11 rank 35053)
Lens window: L5:[ now](3203) L7:[ also](2423) L9:[ kidnapped](1858) L11:[ mentor](50)
Lens top-3 per layer: L5: now, also, still; L7: also, kidnapped, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](736) L7:[ a](514) L9:[ the](787) L11:[SPONSORED](35053)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, a, in, one, also; L11: SPONSORED, soDeliveryDate, enegger, Reviewer, ofi
Generated (8): "the mentor of Elif Rossi. Ruth" — correct
Name in 50-token generation: yes

#### 15. Nathaniel Strand (mentor: Odd Mikkelsen)
Prompt: "Odd Mikkelsen's apprentice is"
Prompt tokens: ['O', 'dd', ' Mik', 'kel', 'sen', "'s", ' apprentice', ' is']
Expected first piece: [Nathaniel]
Verdict: **WEAK** (lens L11 rank 53; cosine L11 rank 812)
Lens window: L5:[ now](6130) L7:[ also](1198) L9:[ kidnapped](1103) L11:[ Ruth](53)
Lens top-3 per layer: L5: now, also, still; L7: also, now, still; L9: kidnapped, murdered, named; L11: Ruth, mentor, T
Cosine window: L5:[ a](27391) L7:[ a](15396) L9:[ the](14807) L11:[SPONSORED](812)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, enegger, Reviewer
Generated (8): "the mentor of Ravi Gale. Gale" — wrong
Name in 50-token generation: no

#### 16. Lars Quill (mentor: Anya Lund)
Prompt: "Anya Lund's apprentice is"
Prompt tokens: ['An', 'ya', ' Lund', "'s", ' apprentice', ' is']
Expected first piece: [Lars]
Verdict: **WEAK** (lens L11 rank 47; cosine L11 rank 64)
Lens window: L5:[ now](621) L7:[ also](541) L9:[ kidnapped](159) L11:[ Ruth](47)
Lens top-3 per layer: L5: now, still, currently; L7: also, now, still; L9: kidnapped, murdered, gunned; L11: Ruth, mentor, T
Cosine window: L5:[ a](3151) L7:[ a](3334) L9:[ the](1440) L11:[SPONSORED](64)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, in; L9: the, a, in, one, an; L11: SPONSORED, ofi, Reviewer, soDeliveryDate, enegger
Generated (8): "the mentor of Lars Quill. Qu" — correct
Name in 50-token generation: yes

#### 17. Erik Lund (mentor: Liva Johansen)
Prompt: "Liva Johansen's apprentice is"
Prompt tokens: ['L', 'iva', ' Joh', 'ansen', "'s", ' apprentice', ' is']
Expected first piece: [Erik]
Verdict: **WEAK** (lens L11 rank 49; cosine L11 rank 2837)
Lens window: L5:[ now](1171) L7:[ also](613) L9:[ missing](78) L11:[ mentor](49)
Lens top-3 per layer: L5: now, also, still; L7: also, now, currently; L9: missing, poised, currently; L11: mentor, Ruth, T
Cosine window: L5:[ a](4951) L7:[ a](3454) L9:[ the](1057) L11:[SPONSORED](2837)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, not, an, also; L9: the, a, in, on, one; L11: SPONSORED, Reviewer, lihood, soDeliveryDate, enegger
Generated (8): "the mentor of Erik Lund. Yil" — correct
Name in 50-token generation: yes

#### 18. Felix Marsh (mentor: Theo Linden)
Prompt: "Theo Linden's apprentice is"
Prompt tokens: ['The', 'o', ' Lind', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Felix]
Verdict: **STORED** (lens L11 rank 10; cosine L11 rank 161)
Lens window: L5:[ now](575) L7:[ also](137) L9:[ named](17) L11:[ Ruth](10)
Lens top-3 per layer: L5: now, also, still; L7: also, now, still; L9: named, murdered, kidnapped; L11: Ruth, mentor, T
Cosine window: L5:[ a](4868) L7:[ a](2628) L9:[ the](1201) L11:[SPONSORED](161)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, also; L9: the, a, in, one, now; L11: SPONSORED, ofi, Hallow, soDeliveryDate, Reviewer
Generated (8): "the mentor of Felix Marsh. Felix is" — correct
Name in 50-token generation: yes

#### 19. Ravi Gale (mentor: Folke Fenwick)
Prompt: "Folke Fenwick's apprentice is"
Prompt tokens: ['F', 'ol', 'ke', ' Fen', 'wick', "'s", ' apprentice', ' is']
Expected first piece: [R]
Verdict: **WEAK** (lens L11 rank 59; cosine L11 rank 46267)
Lens window: L5:[ now](1424) L7:[ also](2758) L9:[ missing](1605) L11:[ mentor](59)
Lens top-3 per layer: L5: now, still, also; L7: also, now, named; L9: missing, murdered, kidnapped; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](57) L7:[ a](79) L9:[ the](102) L11:[SPONSORED](46267)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, enegger, soDeliveryDate, Reviewer
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 20. Cornelia Hagen (mentor: Greta Moen)
Prompt: "Greta Moen's apprentice is"
Prompt tokens: ['G', 'ret', 'a', ' Mo', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Corn]
Verdict: **WEAK** (lens L11 rank 23; cosine L11 rank 6027)
Lens window: L5:[ now](11718) L7:[ also](9873) L9:[ murdered](4858) L11:[ mentor](23)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: murdered, kidnapped, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](10324) L7:[ a](9833) L9:[ the](12492) L11:[SPONSORED](6027)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, also, one; L11: SPONSORED, soDeliveryDate, ofi, enegger, Reviewer
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 21. Marcus Novak (mentor: Asta Tamarack)
Prompt: "Asta Tamarack's apprentice is"
Prompt tokens: ['Ast', 'a', ' Tam', 'ar', 'ack', "'s", ' apprentice', ' is']
Expected first piece: [Marcus]
Verdict: **MISSING** (lens L11 rank 173; cosine L11 rank 21200)
Lens window: L5:[ now](1555) L7:[ also](573) L9:[ murdered](726) L11:[ mentor](173)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: murdered, killed, kidnapped; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](5757) L7:[ a](2955) L9:[ the](2435) L11:[SPONSORED](21200)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, soDeliveryDate, ofi, Reviewer, enegger
Generated (8): "the mentor of Marcus Novak. Nov" — correct
Name in 50-token generation: yes

#### 22. Dmitri Maple (mentor: Elsa Alder)
Prompt: "Elsa Alder's apprentice is"
Prompt tokens: ['Elsa', ' A', 'lder', "'s", ' apprentice', ' is']
Expected first piece: [Dmit]
Verdict: **WEAK** (lens L11 rank 12; cosine L11 rank 64)
Lens window: L5:[ now](30634) L7:[ still](13052) L9:[ murdered](4822) L11:[ mentor](12)
Lens top-3 per layer: L5: now, still, currently; L7: still, also, now; L9: murdered, killed, kidnapped; L11: mentor, Ruth, T
Cosine window: L5:[ a](42568) L7:[ a](33416) L9:[ the](21544) L11:[SPONSORED](64)
Cosine top-5 per layer: L5: a, the, not, in, an; L7: a, the, not, in, an; L9: the, in, a, on, not; L11: SPONSORED, enegger, soDeliveryDate, Reviewer, ofi
Generated (8): "the mentor of Dmitri Maple. Dup" — correct
Name in 50-token generation: yes

#### 23. Nora Fischer (mentor: Tara Haddad)
Prompt: "Tara Haddad's apprentice is"
Prompt tokens: ['T', 'ara', ' H', 'add', 'ad', "'s", ' apprentice', ' is']
Expected first piece: [Nora]
Verdict: **WEAK** (lens L11 rank 39; cosine L11 rank 342)
Lens window: L5:[ now](1893) L7:[ also](1939) L9:[ kidnapped](885) L11:[ mentor](39)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, murdered, missing; L11: mentor, Ruth, T
Cosine window: L5:[ a](9238) L7:[ a](9286) L9:[ the](6014) L11:[SPONSORED](342)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, also; L11: SPONSORED, soDeliveryDate, Reviewer, ofi, enegger
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 24. Hedda Bakken (mentor: Jonas Osprey)
Prompt: "Jonas Osprey's apprentice is"
Prompt tokens: ['Jon', 'as', ' O', 'sp', 'rey', "'s", ' apprentice', ' is']
Expected first piece: [H]
Verdict: **MISSING** (lens L11 rank 175; cosine L11 rank 48741)
Lens window: L5:[ now](587) L7:[ also](1077) L9:[ kidnapped](1571) L11:[ mentor](175)
Lens top-3 per layer: L5: now, still, also; L7: also, still, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](61) L7:[ a](55) L9:[ the](95) L11:[SPONSORED](48741)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, enegger, soDeliveryDate, Reviewer, ofi
Generated (8): "the mentor of Dmitri Maple. Dup" — wrong
Name in 50-token generation: no

#### 25. Mei Larsen (mentor: Casper Sorrel)
Prompt: "Casper Sorrel's apprentice is"
Prompt tokens: ['C', 'asper', ' Sor', 'rel', "'s", ' apprentice', ' is']
Expected first piece: [Mei]
Verdict: **WEAK** (lens L11 rank 21; cosine L11 rank 71)
Lens window: L5:[ now](5634) L7:[ also](3672) L9:[ kidnapped](547) L11:[ mentor](21)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, murdered, missing; L11: mentor, Ruth, T
Cosine window: L5:[ a](15266) L7:[ a](13951) L9:[ the](4773) L11:[SPONSORED](71)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: the, a, in, one, also; L11: SPONSORED, ofi, Reviewer, soDeliveryDate, enegger
Generated (8): "the mentor of Mei Larsen. Mei" — correct
Name in 50-token generation: yes

#### 26. Siri Sable (mentor: Nadia Lindqvist)
Prompt: "Nadia Lindqvist's apprentice is"
Prompt tokens: ['N', 'adia', ' Lind', 'qv', 'ist', "'s", ' apprentice', ' is']
Expected first piece: [Siri]
Verdict: **STORED** (lens L11 rank 8; cosine L11 rank 8)
Lens window: L5:[ now](202) L7:[ also](137) L9:[ named](20) L11:[ Ruth](8)
Lens top-3 per layer: L5: now, also, currently; L7: also, now, still; L9: named, missing, gunned; L11: Ruth, mentor, T
Cosine window: L5:[ a](2985) L7:[ a](3871) L9:[ the](2712) L11:[SPONSORED](8)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, an, not; L9: the, a, in, also, one; L11: SPONSORED, soDeliveryDate, Hallow, ofi, Reviewer
Generated (8): "the mentor of Siri Sable. Siri" — correct
Name in 50-token generation: yes

#### 27. Kirsten Birch (mentor: Anton Sedge)
Prompt: "Anton Sedge's apprentice is"
Prompt tokens: ['Ant', 'on', ' S', 'edge', "'s", ' apprentice', ' is']
Expected first piece: [Kirst]
Verdict: **WEAK** (lens L11 rank 59; cosine L11 rank 514)
Lens window: L5:[ now](17939) L7:[ also](7348) L9:[ kidnapped](2481) L11:[ mentor](59)
Lens top-3 per layer: L5: now, currently, also; L7: also, now, still; L9: kidnapped, murdered, gunned; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](38848) L7:[ a](30903) L9:[ the](21036) L11:[SPONSORED](514)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, Reviewer, enegger, soDeliveryDate
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 28. Rosa Peregrine (mentor: Else Kovac)
Prompt: "Else Kovac's apprentice is"
Prompt tokens: ['Else', ' Kov', 'ac', "'s", ' apprentice', ' is']
Expected first piece: [Rosa]
Verdict: **WEAK** (lens L11 rank 41; cosine L11 rank 269)
Lens window: L5:[ now](4820) L7:[ still](4093) L9:[ kidnapped](1572) L11:[ mentor](41)
Lens top-3 per layer: L5: now, still, not; L7: still, also, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](12266) L7:[ a](12819) L9:[ the](5907) L11:[SPONSORED](269)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, in, a, not, also; L11: SPONSORED, Reviewer, enegger, soDeliveryDate, ofi
Generated (8): "the mentor of Rosa Peregrine." — correct
Name in 50-token generation: yes

#### 29. Mona Fern (mentor: Jette Knot)
Prompt: "Jette Knot's apprentice is"
Prompt tokens: ['J', 'ette', ' Knot', "'s", ' apprentice', ' is']
Expected first piece: [Mon]
Verdict: **WEAK** (lens L11 rank 60; cosine L11 rank 12161)
Lens window: L5:[ now](2617) L7:[ also](5548) L9:[ named](754) L11:[ Ruth](60)
Lens top-3 per layer: L5: now, also, currently; L7: also, named, now; L9: named, kidnapped, missing; L11: Ruth, mentor, T
Cosine window: L5:[ a](590) L7:[ a](1226) L9:[ the](533) L11:[SPONSORED](12161)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, one, an; L11: SPONSORED, soDeliveryDate, Hallow, ofi, Dmit
Generated (8): "the mentor of Jorunn Blackwood" — wrong
Name in 50-token generation: no

#### 30. Amara Yarrow (mentor: Aisha Halvorsen)
Prompt: "Aisha Halvorsen's apprentice is"
Prompt tokens: ['A', 'isha', ' Hal', 'v', 'ors', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Am]
Verdict: **WEAK** (lens L11 rank 27; cosine L11 rank 16692)
Lens window: L5:[ now](925) L7:[ also](712) L9:[ murdered](305) L11:[ mentor](27)
Lens top-3 per layer: L5: now, still, also; L7: also, still, currently; L9: murdered, kidnapped, gunned; L11: mentor, Ruth, T
Cosine window: L5:[ a](411) L7:[ a](275) L9:[ a](300) L11:[SPONSORED](16692)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, in; L9: a, the, in, one, an; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, Hallow
Generated (8): "the mentor of Amara Yarrow." — correct
Name in 50-token generation: yes

#### 31. Gunda Frost (mentor: Ruben Juniper)
Prompt: "Ruben Juniper's apprentice is"
Prompt tokens: ['Rub', 'en', ' Jun', 'iper', "'s", ' apprentice', ' is']
Expected first piece: [Gund]
Verdict: **WEAK** (lens L11 rank 39; cosine L11 rank 25)
Lens window: L5:[ now](9560) L7:[ also](14782) L9:[ murdered](6613) L11:[ Ruth](39)
Lens top-3 per layer: L5: now, currently, still; L7: also, still, currently; L9: murdered, missing, gunned; L11: Ruth, mentor, Cedar
Cosine window: L5:[ a](11510) L7:[ a](15232) L9:[ the](15488) L11:[SPONSORED](25)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, an; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 32. Yuki Umber (mentor: Dagmar Vale)
Prompt: "Dagmar Vale's apprentice is"
Prompt tokens: ['D', 'ag', 'mar', ' Vale', "'s", ' apprentice', ' is']
Expected first piece: [Y]
Verdict: **WEAK** (lens L11 rank 30; cosine L11 rank 40744)
Lens window: L5:[ now](909) L7:[ also](1835) L9:[ kidnapped](1930) L11:[ mentor](30)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](219) L7:[ a](317) L9:[ the](696) L11:[SPONSORED](40744)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, enegger, Reviewer
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 33. Ingeborg Okafor (mentor: Omar Lindgren)
Prompt: "Omar Lindgren's apprentice is"
Prompt tokens: ['O', 'mar', ' Lind', 'gren', "'s", ' apprentice', ' is']
Expected first piece: [In]
Verdict: **WEAK** (lens L11 rank 34; cosine L11 rank 48995)
Lens window: L5:[ now](5783) L7:[ also](4311) L9:[ named](542) L11:[ Ruth](34)
Lens top-3 per layer: L5: now, currently, also; L7: also, now, currently; L9: named, missing, gunned; L11: Ruth, mentor, T
Cosine window: L5:[ a](273) L7:[ a](212) L9:[ the](100) L11:[SPONSORED](48995)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, in; L9: the, a, in, one, on; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, ��
Generated (8): "the mentor of Siri Sable. Siri" — wrong
Name in 50-token generation: no

#### 34. Pia Virtanen (mentor: Farid Lark)
Prompt: "Farid Lark's apprentice is"
Prompt tokens: ['Far', 'id', ' L', 'ark', "'s", ' apprentice', ' is']
Expected first piece: [P]
Verdict: **WEAK** (lens L11 rank 100; cosine L11 rank 48940)
Lens window: L5:[ now](2718) L7:[ also](2797) L9:[ murdered](2170) L11:[ mentor](100)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: murdered, kidnapped, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](104) L7:[ a](77) L9:[ the](110) L11:[SPONSORED](48940)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 35. Walter Elm (mentor: Adele Tarn)
Prompt: "Adele Tarn's apprentice is"
Prompt tokens: ['Ad', 'ele', ' T', 'arn', "'s", ' apprentice', ' is']
Expected first piece: [Walter]
Verdict: **WEAK** (lens L11 rank 22; cosine L11 rank 7068)
Lens window: L5:[ now](2334) L7:[ also](2424) L9:[ kidnapped](1492) L11:[ mentor](22)
Lens top-3 per layer: L5: now, also, currently; L7: also, still, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](4971) L7:[ a](4486) L9:[ the](3786) L11:[SPONSORED](7068)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, also; L9: the, a, in, one, also; L11: SPONSORED, soDeliveryDate, ofi, Reviewer, enegger
Generated (8): "the mentor of Walter Elm. Halloway" — correct
Name in 50-token generation: yes

#### 36. Kofi Kapoor (mentor: Ulrik Flint)
Prompt: "Ulrik Flint's apprentice is"
Prompt tokens: ['Ul', 'rik', ' Flint', "'s", ' apprentice', ' is']
Expected first piece: [K]
Verdict: **MISSING** (lens L11 rank 104; cosine L11 rank 47270)
Lens window: L5:[ now](1508) L7:[ also](3725) L9:[ murdered](2201) L11:[ mentor](104)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: murdered, kidnapped, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](114) L7:[ a](184) L9:[ the](212) L11:[SPONSORED](47270)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, also; L9: the, a, in, one, on; L11: SPONSORED, enegger, Reviewer, soDeliveryDate, ofi
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 37. Ruth Novak (mentor: Ines Ember)
Prompt: "Ines Ember's apprentice is"
Prompt tokens: ['In', 'es', ' Ember', "'s", ' apprentice', ' is']
Expected first piece: [Ruth]
Verdict: **STORED** (lens L11 rank 2; cosine L11 rank 143)
Lens window: L5:[ now](2531) L7:[ also](2290) L9:[ named](499) L11:[ mentor](2)
Lens top-3 per layer: L5: now, currently, still; L7: also, now, still; L9: named, missing, now; L11: mentor, Ruth, T
Cosine window: L5:[ a](9633) L7:[ a](10321) L9:[ the](7726) L11:[SPONSORED](143)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, not, an, also; L9: the, a, in, one, an; L11: SPONSORED, ofi, Hallow, soDeliveryDate, Reviewer
Generated (8): "the mentor of Ruth Novak. Ruth" — correct
Name in 50-token generation: yes

#### 38. Ulf Kone (mentor: Liv Patel)
Prompt: "Liv Patel's apprentice is"
Prompt tokens: ['L', 'iv', ' Patel', "'s", ' apprentice', ' is']
Expected first piece: [U]
Verdict: **WEAK** (lens L11 rank 16; cosine L11 rank 40393)
Lens window: L5:[ now](3594) L7:[ also](4035) L9:[ missing](1298) L11:[ mentor](16)
Lens top-3 per layer: L5: now, also, currently; L7: also, currently, now; L9: missing, gunned, murdered; L11: mentor, Ruth, T
Cosine window: L5:[ a](246) L7:[ a](274) L9:[ the](345) L11:[SPONSORED](40393)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, also, in; L9: the, a, in, one, on; L11: SPONSORED, soDeliveryDate, enegger, Reviewer, lihood
Generated (8): "the mentor of Ulf Kone." — correct
Name in 50-token generation: yes

#### 39. Ragnvald Vesper (mentor: Svend Larsen)
Prompt: "Svend Larsen's apprentice is"
Prompt tokens: ['S', 'v', 'end', ' Lars', 'en', "'s", ' apprentice', ' is']
Expected first piece: [R]
Verdict: **WEAK** (lens L11 rank 52; cosine L11 rank 47308)
Lens window: L5:[ now](2055) L7:[ also](4393) L9:[ kidnapped](2934) L11:[ mentor](52)
Lens top-3 per layer: L5: now, currently, still; L7: also, still, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](70) L7:[ a](102) L9:[ the](181) L11:[SPONSORED](47308)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, a, in, one, on; L11: SPONSORED, enegger, soDeliveryDate, Reviewer, ofi
Generated (8): "the mentor of Mei Larsen. Mei" — wrong
Name in 50-token generation: no

#### 40. Njord Quartz (mentor: Rikke Lie)
Prompt: "Rikke Lie's apprentice is"
Prompt tokens: ['R', 'ik', 'ke', ' Lie', "'s", ' apprentice', ' is']
Expected first piece: [N]
Verdict: **WEAK** (lens L11 rank 43; cosine L11 rank 40749)
Lens window: L5:[ now](929) L7:[ also](1189) L9:[ missing](533) L11:[ Ruth](43)
Lens top-3 per layer: L5: now, still, currently; L7: also, now, still; L9: missing, kidnapped, murdered; L11: Ruth, mentor, Cedar
Cosine window: L5:[ a](111) L7:[ a](82) L9:[ the](85) L11:[SPONSORED](40749)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, ��, Hallow
Generated (8): "the mentor of Mei Larsen. Mei" — wrong
Name in 50-token generation: no

#### 41. Sten Haugen (mentor: Astrid Ashford)
Prompt: "Astrid Ashford's apprentice is"
Prompt tokens: ['A', 'str', 'id', ' Ash', 'ford', "'s", ' apprentice', ' is']
Expected first piece: [St]
Verdict: **WEAK** (lens L11 rank 76; cosine L11 rank 47011)
Lens window: L5:[ now](1600) L7:[ also](4103) L9:[ kidnapped](3388) L11:[ mentor](76)
Lens top-3 per layer: L5: now, still, also; L7: also, suspended, now; L9: kidnapped, murdered, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](169) L7:[ a](256) L9:[ the](636) L11:[SPONSORED](47011)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, a, in, one, also; L11: SPONSORED, soDeliveryDate, enegger, ofi, Reviewer
Generated (8): "the mentor of Mei Larsen. Mei" — wrong
Name in 50-token generation: no

#### 42. Leif Dahl (mentor: Kenji Ulmar)
Prompt: "Kenji Ulmar's apprentice is"
Prompt tokens: ['Ken', 'ji', ' Ul', 'mar', "'s", ' apprentice', ' is']
Expected first piece: [Le]
Verdict: **WEAK** (lens L11 rank 81; cosine L11 rank 39578)
Lens window: L5:[ now](4795) L7:[ also](4957) L9:[ kidnapped](969) L11:[ mentor](81)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, killed, murdered; L11: mentor, Ruth, T
Cosine window: L5:[ a](842) L7:[ a](833) L9:[ the](300) L11:[SPONSORED](39578)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, enegger, Reviewer, soDeliveryDate, ofi
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 43. Pavel Hagen (mentor: Harald Haga)
Prompt: "Harald Haga's apprentice is"
Prompt tokens: ['Har', 'ald', ' H', 'aga', "'s", ' apprentice', ' is']
Expected first piece: [Pavel]
Verdict: **WEAK** (lens L11 rank 24; cosine L11 rank 109)
Lens window: L5:[ now](17461) L7:[ also](5013) L9:[ murdered](1619) L11:[ mentor](24)
Lens top-3 per layer: L5: now, still, also; L7: also, now, still; L9: murdered, kidnapped, killed; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](41583) L7:[ a](27991) L9:[ the](15493) L11:[SPONSORED](109)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, one, not; L11: SPONSORED, ofi, Reviewer, enegger, soDeliveryDate
Generated (8): "the mentor of Pavel Hagen. Pavel" — correct
Name in 50-token generation: yes

#### 44. Birgit Dupont (mentor: Frida Meyer)
Prompt: "Frida Meyer's apprentice is"
Prompt tokens: ['Fr', 'ida', ' Meyer', "'s", ' apprentice', ' is']
Expected first piece: [Bir]
Verdict: **WEAK** (lens L11 rank 64; cosine L11 rank 1292)
Lens window: L5:[ now](5366) L7:[ also](2058) L9:[ murdered](883) L11:[ mentor](64)
Lens top-3 per layer: L5: now, also, still; L7: also, now, still; L9: murdered, kidnapped, gunned; L11: mentor, Ruth, T
Cosine window: L5:[ a](6040) L7:[ a](3448) L9:[ the](2458) L11:[SPONSORED](1292)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, an; L9: the, a, in, one, on; L11: SPONSORED, soDeliveryDate, enegger, ofi, Reviewer
Generated (8): "the mentor of Otto Zephyr." — wrong
Name in 50-token generation: no

#### 45. Palle Drift (mentor: Tomas Marlowe)
Prompt: "Tomas Marlowe's apprentice is"
Prompt tokens: ['Tom', 'as', ' Mar', 'low', 'e', "'s", ' apprentice', ' is']
Expected first piece: [Pal]
Verdict: **WEAK** (lens L11 rank 67; cosine L11 rank 10605)
Lens window: L5:[ now](2312) L7:[ also](4142) L9:[ kidnapped](1219) L11:[ mentor](67)
Lens top-3 per layer: L5: now, still, also; L7: also, now, still; L9: kidnapped, murdered, killed; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](1917) L7:[ a](3010) L9:[ the](2026) L11:[SPONSORED](10605)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, in; L9: the, a, in, one, not; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 46. Jussi Petrov (mentor: Trine Mendez)
Prompt: "Trine Mendez's apprentice is"
Prompt tokens: ['Tr', 'ine', ' Mend', 'ez', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **WEAK** (lens L11 rank 61; cosine L11 rank 45909)
Lens window: L5:[ now](1021) L7:[ also](1997) L9:[ gunned](1898) L11:[ mentor](61)
Lens top-3 per layer: L5: now, still, also; L7: also, still, now; L9: gunned, murdered, kidnapped; L11: mentor, Ruth, T
Cosine window: L5:[ a](133) L7:[ a](135) L9:[ the](283) L11:[SPONSORED](45909)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, not; L11: SPONSORED, enegger, soDeliveryDate, ofi, Reviewer
Generated (8): "the mentor of Dante Sato. Dante" — wrong
Name in 50-token generation: no

#### 47. Otto Zephyr (mentor: Morten Wren)
Prompt: "Morten Wren's apprentice is"
Prompt tokens: ['M', 'ort', 'en', ' W', 'ren', "'s", ' apprentice', ' is']
Expected first piece: [Otto]
Verdict: **WEAK** (lens L11 rank 51; cosine L11 rank 125)
Lens window: L5:[ now](99) L7:[ also](102) L9:[ murdered](33) L11:[ mentor](51)
Lens top-3 per layer: L5: now, currently, still; L7: also, still, now; L9: murdered, killed, kidnapped; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](1976) L7:[ a](2060) L9:[ the](1005) L11:[SPONSORED](125)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, also, one; L11: SPONSORED, ofi, enegger, Reviewer, ��
Generated (8): "the mentor of Otto Zephyr." — correct
Name in 50-token generation: yes

#### 48. Jorunn Blackwood (mentor: Yara Dunmore)
Prompt: "Yara Dunmore's apprentice is"
Prompt tokens: ['Y', 'ara', ' Dun', 'more', "'s", ' apprentice', ' is']
Expected first piece: [J]
Verdict: **WEAK** (lens L11 rank 57; cosine L11 rank 43796)
Lens window: L5:[ now](859) L7:[ also](924) L9:[ kidnapped](821) L11:[ mentor](57)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, murdered, missing; L11: mentor, Ruth, T
Cosine window: L5:[ a](118) L7:[ a](96) L9:[ the](129) L11:[SPONSORED](43796)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, one, not; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Yuki Umber." — wrong
Name in 50-token generation: no

#### 49. Gitte Tanaka (mentor: Carlos Nettle)
Prompt: "Carlos Nettle's apprentice is"
Prompt tokens: ['Carl', 'os', ' N', 'ettle', "'s", ' apprentice', ' is']
Expected first piece: [G]
Verdict: **MISSING** (lens L11 rank 151; cosine L11 rank 48486)
Lens window: L5:[ now](1163) L7:[ also](2581) L9:[ kidnapped](2922) L11:[ mentor](151)
Lens top-3 per layer: L5: now, also, currently; L7: also, now, still; L9: kidnapped, gunned, killed; L11: mentor, Ruth, T
Cosine window: L5:[ a](70) L7:[ a](99) L9:[ the](189) L11:[SPONSORED](48486)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, in; L9: the, a, in, one, also; L11: SPONSORED, soDeliveryDate, ofi, enegger, Reviewer
Generated (8): "the mentor of Rosa Peregrine." — wrong
Name in 50-token generation: no

#### 50. Mira Mensah (mentor: Stig Solheim)
Prompt: "Stig Solheim's apprentice is"
Prompt tokens: ['St', 'ig', ' Sol', 'heim', "'s", ' apprentice', ' is']
Expected first piece: [Mir]
Verdict: **WEAK** (lens L11 rank 96; cosine L11 rank 8243)
Lens window: L5:[ now](1781) L7:[ also](2047) L9:[ missing](888) L11:[ Ruth](96)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: missing, murdered, kidnapped; L11: Ruth, mentor, Cedar
Cosine window: L5:[ a](2624) L7:[ a](2741) L9:[ the](2243) L11:[SPONSORED](8243)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, also, an; L9: the, a, in, one, not; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Ragnvald Ves" — wrong
Name in 50-token generation: no

#### 51. Celia Moss (mentor: Emil Rahman)
Prompt: "Emil Rahman's apprentice is"
Prompt tokens: ['Em', 'il', ' Rahman', "'s", ' apprentice', ' is']
Expected first piece: [Cel]
Verdict: **WEAK** (lens L11 rank 16; cosine L11 rank 739)
Lens window: L5:[ now](2999) L7:[ also](1573) L9:[ missing](449) L11:[ mentor](16)
Lens top-3 per layer: L5: now, also, still; L7: also, now, still; L9: missing, gunned, kidnapped; L11: mentor, Ruth, T
Cosine window: L5:[ a](4285) L7:[ a](3372) L9:[ a](2602) L11:[SPONSORED](739)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, an, in; L9: a, the, in, one, now; L11: SPONSORED, soDeliveryDate, ofi, enegger, Reviewer
Generated (8): "the mentor of Mei Larsen. Mei" — wrong
Name in 50-token generation: no

#### 52. Dante Sato (mentor: Quinn Sharma)
Prompt: "Quinn Sharma's apprentice is"
Prompt tokens: ['Qu', 'inn', ' Sharma', "'s", ' apprentice', ' is']
Expected first piece: [Dante]
Verdict: **WEAK** (lens L11 rank 17; cosine L11 rank 47)
Lens window: L5:[ now](411) L7:[ also](154) L9:[ kidnapped](54) L11:[ mentor](17)
Lens top-3 per layer: L5: now, also, currently; L7: also, now, currently; L9: kidnapped, missing, named; L11: mentor, Ruth, T
Cosine window: L5:[ a](2386) L7:[ a](2158) L9:[ the](1445) L11:[SPONSORED](47)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, an, not; L9: the, a, in, one, also; L11: SPONSORED, soDeliveryDate, enegger, ofi, Reviewer
Generated (8): "the mentor of Mei Larsen. Mei" — wrong
Name in 50-token generation: yes

#### 53. Tilda Hazel (mentor: Hana Cinder)
Prompt: "Hana Cinder's apprentice is"
Prompt tokens: ['H', 'ana', ' C', 'inder', "'s", ' apprentice', ' is']
Expected first piece: [T]
Verdict: **STORED** (lens L11 rank 3; cosine L11 rank 40856)
Lens window: L5:[ now](537) L7:[ also](1375) L9:[ kidnapped](510) L11:[ mentor](3)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, murdered, gunned; L11: mentor, Ruth, T
Cosine window: L5:[ a](38) L7:[ a](65) L9:[ the](59) L11:[SPONSORED](40856)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, an, not, in; L9: the, a, in, one, not; L11: SPONSORED, ofi, Reviewer, soDeliveryDate, enegger
Generated (8): "the mentor of Walter Elm. Halloway" — wrong
Name in 50-token generation: no

#### 54. Vera Khan (mentor: Peder Jasper)
Prompt: "Peder Jasper's apprentice is"
Prompt tokens: ['P', 'eder', ' Jasper', "'s", ' apprentice', ' is']
Expected first piece: [Vera]
Verdict: **WEAK** (lens L11 rank 37; cosine L11 rank 329)
Lens window: L5:[ now](4306) L7:[ also](5817) L9:[ kidnapped](6419) L11:[ mentor](37)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: kidnapped, missing, murdered; L11: mentor, Ruth, T
Cosine window: L5:[ a](11190) L7:[ a](16032) L9:[ the](21557) L11:[SPONSORED](329)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, an; L9: the, a, in, on, one; L11: SPONSORED, ofi, soDeliveryDate, enegger, Reviewer
Generated (8): "the mentor of Vera Khan. Dupont" — correct
Name in 50-token generation: yes

#### 55. Samir Brass (mentor: Sigrid Cedar)
Prompt: "Sigrid Cedar's apprentice is"
Prompt tokens: ['S', 'igr', 'id', ' Cedar', "'s", ' apprentice', ' is']
Expected first piece: [Sam]
Verdict: **WEAK** (lens L11 rank 51; cosine L11 rank 30011)
Lens window: L5:[ now](332) L7:[ also](491) L9:[ missing](253) L11:[ mentor](51)
Lens top-3 per layer: L5: now, currently, still; L7: also, now, still; L9: missing, gunned, murdered; L11: mentor, Ruth, T
Cosine window: L5:[ a](304) L7:[ a](374) L9:[ the](333) L11:[SPONSORED](30011)
Cosine top-5 per layer: L5: a, the, an, not, now; L7: a, the, not, in, also; L9: the, a, in, one, on; L11: SPONSORED, soDeliveryDate, Reviewer, ofi, enegger
Generated (8): "the mentor of Ruth Novak. Ruth" — wrong
Name in 50-token generation: no

#### 56. Lotte Quarry (mentor: Henrik Dahl)
Prompt: "Henrik Dahl's apprentice is"
Prompt tokens: ['Hen', 'rik', ' Dahl', "'s", ' apprentice', ' is']
Expected first piece: [L]
Verdict: **WEAK** (lens L11 rank 32; cosine L11 rank 43656)
Lens window: L5:[ now](1734) L7:[ also](3007) L9:[ kidnapped](528) L11:[ mentor](32)
Lens top-3 per layer: L5: now, also, still; L7: also, named, still; L9: kidnapped, murdered, killed; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](121) L7:[ a](126) L9:[ the](63) L11:[SPONSORED](43656)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, in, not, also; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, enegger
Generated (8): "the mentor of Mei Larsen. Dahl" — wrong
Name in 50-token generation: no

#### 57. Sana Bakker (mentor: Torvald Larsen)
Prompt: "Torvald Larsen's apprentice is"
Prompt tokens: ['Tor', 'v', 'ald', ' Lars', 'en', "'s", ' apprentice', ' is']
Expected first piece: [Sana]
Verdict: **WEAK** (lens L11 rank 62; cosine L11 rank 237)
Lens window: L5:[ now](3416) L7:[ also](5856) L9:[ murdered](3569) L11:[ mentor](62)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: murdered, kidnapped, killed; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](21590) L7:[ a](26168) L9:[ the](22944) L11:[SPONSORED](237)
Cosine top-5 per layer: L5: a, the, an, in, not; L7: a, the, not, in, an; L9: the, a, in, one, on; L11: SPONSORED, ofi, Reviewer, soDeliveryDate, enegger
Generated (8): "the mentor of Vera Khan. Halloway" — wrong
Name in 50-token generation: no

#### 58. Tove Lindholm (mentor: Noor Zephyrine)
Prompt: "Noor Zephyrine's apprentice is"
Prompt tokens: ['N', 'oor', ' Z', 'eph', 'yr', 'ine', "'s", ' apprentice', ' is']
Expected first piece: [T]
Verdict: **STORED** (lens L11 rank 4; cosine L11 rank 41381)
Lens window: L5:[ now](390) L7:[ also](1108) L9:[ murdered](604) L11:[ mentor](4)
Lens top-3 per layer: L5: now, currently, still; L7: also, now, still; L9: murdered, missing, kidnapped; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](35) L7:[ a](43) L9:[ the](61) L11:[SPONSORED](41381)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, not, in, also; L9: the, a, in, one, also; L11: SPONSORED, ofi, soDeliveryDate, Reviewer, Hallow
Generated (8): "the mentor of Noor Yew." — wrong
Name in 50-token generation: no

#### 59. Lena Watanabe (mentor: Ditlev Berg)
Prompt: "Ditlev Berg's apprentice is"
Prompt tokens: ['D', 'itle', 'v', ' Berg', "'s", ' apprentice', ' is']
Expected first piece: [Lena]
Verdict: **WEAK** (lens L11 rank 91; cosine L11 rank 2110)
Lens window: L5:[ now](18103) L7:[ also](5607) L9:[ murdered](1437) L11:[ mentor](91)
Lens top-3 per layer: L5: now, still, currently; L7: also, still, now; L9: murdered, kidnapped, killed; L11: mentor, Ruth, Cedar
Cosine window: L5:[ a](35777) L7:[ a](23228) L9:[ the](9852) L11:[SPONSORED](2110)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, in, not, also; L9: the, a, in, one, on; L11: SPONSORED, ofi, enegger, Reviewer, soDeliveryDate
Generated (8): "the mentor of Dmitri Maple. Dup" — wrong
Name in 50-token generation: no

#### 60. Zara Halloway (mentor: Mads Rossi)
Prompt: "Mads Rossi's apprentice is"
Prompt tokens: ['M', 'ads', ' Rossi', "'s", ' apprentice', ' is']
Expected first piece: [Z]
Verdict: **WEAK** (lens L11 rank 61; cosine L11 rank 40678)
Lens window: L5:[ now](2112) L7:[ also](4036) L9:[ missing](2033) L11:[ mentor](61)
Lens top-3 per layer: L5: now, currently, also; L7: also, now, still; L9: missing, gunned, kidnapped; L11: mentor, Ruth, T
Cosine window: L5:[ a](212) L7:[ a](390) L9:[ the](436) L11:[SPONSORED](40678)
Cosine top-5 per layer: L5: a, the, an, not, in; L7: a, the, also, not, an; L9: the, a, in, also, one; L11: SPONSORED, soDeliveryDate, enegger, Reviewer, ofi
Generated (8): "the mentor of Mona Fern. Ruth" — wrong
Name in 50-token generation: no
