# 경제 뉴스 수집 검증 리포트

기간 2014-01-01 ~ 2025-12-31 (4383일). 생성: `collect_economic_news.py --verify`

## US — `공통데이터/USD/US_economic_news_daily.csv`

- 총 행 14958, 고유 날짜 **4383 / 4383**, 일평균 3.41건

| 검사 | 결과 |
|---|---|
| 모든 날짜에 ≥1건 (누락 0일) | PASS |
| 범위 밖 날짜 행 0 | PASS |
| 날짜+헤드라인 중복 0 | PASS |
| 빈 헤드라인 0 | PASS |
| year/month 불일치 0 | PASS |
| 장 마감 요약 패턴 행(선택본, fallback 제외) 0 | PASS |
| URL 형식 오류 0 | PASS |

| 연도 | 커버 일수 | 행 수 | 일평균 | fallback 일수 | fallback 비율 |
|---|---|---|---|---|---|
| 2014 | 365 / 365 | 1318 | 3.61 | 31 | 8.5% |
| 2015 | 365 / 365 | 2103 | 5.76 | 18 | 4.9% |
| 2016 | 366 / 366 | 1665 | 4.55 | 18 | 4.9% |
| 2017 | 365 / 365 | 1135 | 3.11 | 37 | 10.1% |
| 2018 | 365 / 365 | 1005 | 2.75 | 53 | 14.5% |
| 2019 | 365 / 365 | 907 | 2.48 | 65 | 17.8% |
| 2020 | 366 / 366 | 1160 | 3.17 | 48 | 13.1% |
| 2021 | 365 / 365 | 1144 | 3.13 | 63 | 17.3% |
| 2022 | 365 / 365 | 1303 | 3.57 | 46 | 12.6% |
| 2023 | 365 / 365 | 1056 | 2.89 | 54 | 14.8% |
| 2024 | 366 / 366 | 940 | 2.57 | 70 | 19.1% |
| 2025 | 365 / 365 | 1222 | 3.35 | 48 | 13.2% |

### 약한 날짜 (fallback 이면서 관련도 점수 ≤ 1) — 176일

해당 매체에 그날 경제 뉴스가 사실상 없어 그날 최고점 기사를 유지한 경우. 필요하면 학습에서 제외하거나 가중치를 낮출 것.

- 2014-03-30 [business/international, score 1] European Lawmakers Prepare to Vote on ‘Net Neutrality’
- 2014-05-11 [business/international, score 1] Scottish Independence Vote Could Set Off E.U. Exodus
- 2014-05-17 [business/international, score 1] Slowly Producing a Spirit England Can Call Its Own
- 2014-05-25 [upshot, score 1] When Hospital Systems Buy Health Insurers
- 2014-06-15 [business/media, score 1] Casey Kasem, Wholesome Voice of Pop Radio, Dies at 82
- 2014-11-23 [business, score 1] Chrysler Details Its Plans to Improve Recall Efforts
- 2014-12-14 [business/international, score 1] Greece Is Not Headed for the Door Just Yet
- 2015-04-05 [business/media, score 1] China Escalates Hollywood Partnerships, Aiming to Compete One Day
- 2015-05-02 [business/media, score 1] William Pfaff, Critic of American Foreign Policy, Dies at 86
- 2015-09-19 [business, score 1] In Philadelphia, Putting on a Show for a Holy Headliner
- 2016-01-30 [business, score 1] Unfamiliar Terrain for Corporate Lawyer in Planned Parenthood Drama
- 2016-06-11 [business/media, score 1] Gawker’s Appeal in Sale May Be Its E-Commerce Potential
- 2016-07-16 [business/media, score 1] In Ex-Fox Anchor Harassment Case, Accusations of ‘Judge Shopping’
- 2016-08-13 [business, score 1] Why Some Life Insurance Premiums Are Skyrocketing
- 2016-09-04 [business/international, score 1] Raghuram Rajan, India’s Departing Central Banker, Has a New Warning
- 2016-09-24 [business/international, score 1] In Australia, China’s Appetite Shifts From Rocks to Real Estate
- 2016-10-01 [business/energy-environment, score 1] A Curious Plan to Fight Climate Change: Buy Mines, Sell Coal
- 2016-11-05 [business, score 1] The Money Management Gospel of Yale’s Endowment Guru
- 2016-12-31 [business, score 1] Costly Drug for Fatal Muscular Disease Wins F.D.A. Approval
- 2017-07-15 [business, score 1] Alfred Angelo Bridal Chain Closes. Scrambling Ensues.
- 2017-10-21 [business, score 1] The Finger-Pointing at the Finance Firm TIAA
- 2017-10-22 [business, score 1] A Showdown Brews Between Amazon and Alibaba, Far From Home
- 2017-12-17 [business/media, score 1] Netflix and Spotify Ask: Can Data Mining Make for Cute Ads?
- 2018-01-07 [business, score 1] Kushner’s Financial Ties to Israel Deepen Even With Mideast Diplomatic Role
- 2018-01-14 [business, score 1] False Missile Warning in Hawaii Adds to Scrutiny of Emergency Alert System
- 2018-01-21 [business, score 1] Nations Seek the Elusive Cure for Cyberattacks
- 2018-03-17 [business, score 1] Catharine MacKinnon and Gretchen Carlson Have a Few Things to Say
- 2018-03-31 [business, score 1] Tesla Says Crashed Vehicle Had Been on Autopilot Before Fatal Accident
- 2018-04-21 [business, score 1] With a Glance Backward, Brooks Brothers Looks to the Future
- 2018-05-19 [business/media, score 1] Lawsuit Brought by Ex-Fox News Host Andrea Tantaros Is Dismissed
- 2018-06-03 [business/media, score 1] A Sign of ‘Modern Society’: More Multiracial Families in Commercials
- 2018-07-14 [business, score 1] A Crisis Management Guru Bungles a Crisis
- 2018-07-21 [business/media, score 1] Ta-Nehisi Coates Is Leaving The Atlantic
- 2018-09-30 [business, score 1] First Day for Goldman’s New C.E.O., and Job Numbers Come Out
- 2018-10-06 [upshot, score 1] The Trump Trade Strategy Is Coming Into Focus. That Doesn’t Necessarily Mean It Will Work.
- 2018-10-20 [business/energy-environment, score 1] Something New May Be Rising Off California Coast: Wind Farms
- 2018-11-17 [business/energy-environment, score 1] California Utility Gets Reassurance on Wildfire Liability
- 2018-11-24 [business, score 1] Marijuana Legalization Threatens These Dogs’ Collars
- 2018-12-02 [business, score 1] This Week in Business: G.M. Idles Plants, and Sundar Pichai Goes to Washington
- 2018-12-08 [world/canada, score 1] Justin Trudeau Is Facing a Carbon Tax Backlash. He’s Not Alone.
- 2018-12-09 [business, score 1] The Week in Business: The Emails Facebook Doesn’t Want You to See
- 2018-12-22 [business, score 1] Some Hershey’s Kisses Are Missing Tips and Bakers Want to Know Why
- 2018-12-23 [business, score 1] Carlos Ghosn, Fallen Nissan Chairman, Will Stay in Jail
- 2019-01-05 [business/energy-environment, score 1] As Fires Ravaged California, Utilities Lobbied Lawmakers for Protection
- 2019-01-13 [business/media, score 1] East Bay Express Lays Off Most of Editorial Staff in Latest Blow to Alt-Weeklies
- 2019-01-26 [business, score 1] Why Are Young People Pretending to Love Work?
- 2019-02-09 [business/media, score 1] Jeff Bezos’ Extortion Claim Said to Be Under Review by Prosecutors
- 2019-02-24 [business, score 1] China’s Tech Firms Are Mapping Pig Faces
- 2019-03-03 [business/media, score 1] Michael Jackson Documentary Revives Lurid Claims, Imperiling His Thriving Estate
- 2019-03-23 [business, score 1] Boeing Was ‘Go, Go, Go’ to Beat Airbus With the 737 Max
- 2019-03-30 [business, score 1] The Week in Business: The Sackler Family Gets Sued, and Theresa May’s Darkest Hour
- 2019-04-20 [business, score 1] Tesla to Shrink Board of Directors by Four People
- 2019-04-27 [business, score 1] How China’s ‘Unicorns’ Shook a Bicycle Town
- 2019-04-28 [business/media, score 1] Perspective, Not Parody, at a Subdued Correspondents’ Dinner
- 2019-05-11 [business, score 1] Why Rewards for Loyal Spenders Are ‘a Honey Pot for Hackers’
- 2019-06-02 [business, score 1] China Strikes Defiant Stance on Trade Against Trump
- 2019-07-13 [business, score 1] The Week in Business: Amazon Won’t Let the Robots Take Over
- 2019-07-20 [business, score 1] Equifax Is Said to Be Close to Reaching Deal in Huge ’17 Data Breach
- 2019-07-27 [business, score 1] In an Industrial Corner of France, 18,000 Jobs Are On Offer. Why Aren’t People Taking Them?
- 2019-08-10 [upshot, score 1] On Guns, Public Opinion and Public Policy Often Diverge
- … 외 116일

### 무작위 표본 20일 (날짜별 1순위 헤드라인)

- 2014-07-24 [business/economy] I.M.F. Sees Growth in U.S. as Slowest Since Recession
- 2014-08-06 [business] Investors Punish a Pair of Failed Mergers
- 2014-09-02 [upshot] You Lose, We Win: Consultants Profit Even When Candidates Underperform
- 2014-09-18 [business] Home Prices Decline for 4th Month in China
- 2015-12-14 [business/dealbook] Junk Bond Fund’s Chief Departs After Blocking Withdrawals
- 2016-02-07 [business] A Short Shutdown for Chipotle, Viacom’s Earnings and Yellen to Testify
- 2016-04-19 [business] Wall St. Is Mostly Higher, Helped by Oil Prices
- 2016-07-01 [business/dealbook] Slowdown in Merger Deals Attributed to Political Uncertainty
- 2017-02-17 [business/economy] Trump’s Inroads in Union Ranks Have Labor Leaders Scrambling
- 2018-06-17 [business] Investors Fret About a Trade War, but They Aren’t Fleeing the Stock Market
- 2018-11-27 [business/dealbook] DealBook Briefing: G.M.’s Cuts Spell Trouble for the Economy
- 2018-12-11 [business] Wall St. Gives Up Early Gains as Boost From China-U.S. Trade Talks Fades
- 2019-01-03 [business/dealbook] Bristol-Myers to Acquire Celgene in Deal Worth $74 Billion
- 2019-03-21 [business/dealbook] DealBook Briefing: The Fed Gave Trump What He Wants, Carefully
- 2019-06-30 [business/dealbook] DealBook Special: Wall Street’s Unexpectedly Hot First Half of 2019
- 2020-03-03 [upshot] Is Fiscal Stimulus the Answer to Preventing a Coronavirus Recession?
- 2023-05-30 [business/economy] Companies Push Prices Higher, Protecting Profits but Adding to Inflation
- 2023-06-19 [business/economy] China’s Economic Rebound Hits a Wall, With ‘No Quick Fix’ to Revive It
- 2024-01-28 [business/economy] Private Equity Is Starting to Share With Workers, Without Taking a Financial Hit
- 2025-05-02 [business] US Hiring Stayed Strong Amid Early Days of Tariff Policy, Jobs Report Shows

## UK — `공통데이터/UK/UK_economic_news_daily.csv`

- 총 행 24573, 고유 날짜 **4383 / 4383**, 일평균 5.61건

| 검사 | 결과 |
|---|---|
| 모든 날짜에 ≥1건 (누락 0일) | PASS |
| 범위 밖 날짜 행 0 | PASS |
| 날짜+헤드라인 중복 0 | PASS |
| 빈 헤드라인 0 | PASS |
| year/month 불일치 0 | PASS |
| 장 마감 요약 패턴 행(선택본, fallback 제외) 0 | PASS |
| URL 형식 오류 0 | PASS |

| 연도 | 커버 일수 | 행 수 | 일평균 | fallback 일수 | fallback 비율 |
|---|---|---|---|---|---|
| 2014 | 365 / 365 | 2660 | 7.29 | 24 | 6.6% |
| 2015 | 365 / 365 | 2694 | 7.38 | 6 | 1.6% |
| 2016 | 366 / 366 | 2616 | 7.15 | 4 | 1.1% |
| 2017 | 365 / 365 | 1768 | 4.84 | 10 | 2.7% |
| 2018 | 365 / 365 | 1804 | 4.94 | 16 | 4.4% |
| 2019 | 365 / 365 | 1742 | 4.77 | 17 | 4.7% |
| 2020 | 366 / 366 | 1853 | 5.06 | 15 | 4.1% |
| 2021 | 365 / 365 | 1852 | 5.07 | 6 | 1.6% |
| 2022 | 365 / 365 | 2059 | 5.64 | 2 | 0.5% |
| 2023 | 365 / 365 | 1870 | 5.12 | 12 | 3.3% |
| 2024 | 366 / 366 | 1887 | 5.16 | 8 | 2.2% |
| 2025 | 365 / 365 | 1768 | 4.84 | 14 | 3.8% |

### 약한 날짜 (fallback 이면서 관련도 점수 ≤ 1) — 6일

해당 매체에 그날 경제 뉴스가 사실상 없어 그날 최고점 기사를 유지한 경우. 필요하면 학습에서 제외하거나 가중치를 낮출 것.

- 2014-07-26 [technology, score 0] Is it really so smart to make your finances an open book?
- 2014-09-20 [money, score 1] Is it worth upgrading to the iPhone 6?
- 2023-12-25 [politics, score 1] Labour to crack down on ‘dodgy’ candy stores in push to revive high streets
- 2025-02-16 [global, score 1] Most of UK’s big build-to-rent developers owned by foreign private equity firms
- 2025-02-23 [global, score 1] How UK and EU could find more money for defence without spooking the markets
- 2025-04-26 [politics, score 0] For political parties judging this year’s local election results it’s all about the baseline

### 무작위 표본 20일 (날짜별 1순위 헤드라인)

- 2014-02-23 [business] Housing market anticipates confirmation of recovery
- 2014-12-22 [business] Russian bank gets 30 billion rouble bailout – business live
- 2016-01-30 [business] Rent bills high enough to make any student rebel
- 2016-03-03 [business] UK's record low interest rates to continue in 2016
- 2016-04-17 [business] IMF urges more spending to boost growth
- 2016-10-19 [business] Bank of England rebuts May and Hague's attacks on quantitative easing
- 2017-06-27 [business] IMF cuts US growth forecast, as Bank of England tells banks to boost capital - as it happened
- 2017-07-31 [business] BNP Paribas to expand further into UK property business with merger
- 2018-10-30 [business] No-deal Brexit would trigger lengthy UK recession, warns S&P
- 2019-12-07 [business] Retail landlords fall into a new crisis at Christmas
- 2020-03-26 [business] Record jump in US jobless claims as Covid-19 drives America into recession - as it happened
- 2020-03-28 [business] Complacency to chaos: how Covid-19 sent the world’s markets into freefall
- 2021-07-20 [business] Pound hits five-month low v dollar but markets recover amid Covid-19 worries – as it happened
- 2021-08-19 [business] Lloyds plans big move into UK rental market with 50,000 homes
- 2021-09-18 [business] Government holds urgent talks with energy firms over gas price rises
- 2022-01-19 [business] Burberry predicts 35% rise in annual profits backed by Asia sales
- 2022-06-28 [business] Global energy price cap: market manipulation whose time has come?
- 2022-07-10 [technology] Elon Musk may have to complete $44bn Twitter takeover, legal experts say
- 2023-06-25 [business] Governments advised to cut public spending or raise taxes to curb inflation
- 2024-04-21 [business] Regulators leave Royal Mail vulnerable to ‘corporate predators’, says investor
