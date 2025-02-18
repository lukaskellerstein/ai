# Unicorn College

# Katedra informačních technologií

# Bakalářská práce

# Investování a spekulace na finančním trhu

```
Autor BP: Lukáš Kellerstein
Vedoucí BP: Ing. Petr Hájek Ph.D.
```

```
2012 Praha
```

**Čestné prohlášení**

Prohlašuji, že svou bakalářskou práci na téma Investování a spekulace na finančním trhu jsem
vypracoval samostatně pod vedením vedoucího bakalářské práce a s použitím odborné
literatury a dalších informačních zdrojů, které jsou v práci citovány a jsou též uvedeny v
seznamu literatury a použitých zdrojů.

Jako autor uvedené bakalářské práce dále prohlašuji, že v souvislosti s vytvořením této
bakalářské práce jsem neporušil autorská práva třetích osob, zejména jsem nezasáhl
nedovoleným způsobem do cizích autorských práv osobnostních a jsem si plně vědom
následků porušení ustanovení § 11 a následujících autorského zákona č. 121/2000 Sb.

V Praze dne 3. května 2012 ...................................
Lukáš Kellerstein

**Poděkování**

Děkuji vedoucímu bakalářské práce Ing. Petrovi Hájkovi, Ph.D za účinnou metodickou,
pedagogickou a odbornou pomoc a další cenné rady při zpracování mé bakalářské práce.

**Abstrakt**

V bakalářské práci byly popsány, implementovány, aplikovány a porovnány tři investiční
strategie. Při popisu byla stanovena pravidla investičních strategií a identifikovány potřeby pro
implementaci. Ta probíhala v jazyce C# za pomocí .NET Frameworku 4.0 a statistického
softwaru Wolfram Mathematica 8.0, ve kterém byly provedeny potřebné výpočty. Tržní data
byla získána pomocí software společnosti IQFeed. Aplikace strategií na trhy byla uskutečněna
pomocí Out-of-sample testovaní, díky kterému bylo možno odstranit přefitované strategie. Na
závěr bylo vydáno doporučení pro kombinace trh-strategie k reálnému obchodování.

Klíčová slova: spekulace, investování, strategie, indikátor, riziko, výnos, out-of-sample test

**Abstract**

In the thesis we described, implemented, applied and compared three investment strategies. In
describing the rules we set investment strategies and have identified the need for
implementation. It took place in C # using. NET Framework 4.0, and statistical software
Wolfram Mathematica 8.0, where we performed the necessary calculations. Market data we
obtain using IQFeed software. The application of strategies to markets we have made with
out-of-sample testing, thanks to which we were able to remove přefitované strategy. The
conclusion was a recommendation for a combination of market-strategy for real trading.

Keywords: speculation, investing, strategy, indicator, risk, yield, out-of-sample test

## Obsah

- 1. Úvod
- 2. Teoretická část
  - 2.1 Trhy
    - 2.1.1 ETF
    - 2.1.2 Futures
  - 2.2 Investiční strategie
    - 2.2.1 Tržní strategie
    - 2.2.2 Párové strategie
  - 2.3 Aplikace investiční strategie
    - 2.3.1 Backtest
    - 2.3.2 Optimalizace
    - 2.3.3 Out-of-sample test
  - 2.4 Analýza výsledků
    - 2.4.1 Popisná statistika
    - 2.4.2 Teorie pravděpodobnosti
- 3. Praktická část
  - 3.1 Analýza a návrh
    - 3.1.1 Funkční požadavky
    - 3.1.2 Nefunkční požadavky
    - 3.1.3 Seznam a popis případů užití
    - 3.1.4 High-level pohled na architekturu
  - 3.2 Implementace
    - 3.2.1 Zvolené technologie
    - 3.2.2 Entity prezentační vrstvy
    - 3.2.3 Entity aplikační vrstvy
    - 3.2.4 Entity datové vrstvy
  - 3.3 Aplikace
    - 3.3.1 Trhy
    - 3.3.2 In-sample optimalizace
    - 3.3.3 Out-of-sample backtest
- 4. Zhodnocení výsledků
- 5. Závěr.........................................................................................................................................
- 6. Seznam použitých zdrojů...........................................................................................................
- 7. Seznam zkratek
- 8. Seznam obrázků
- 9. Seznam tabulek
- 10. Seznam Příloh
  - 10.1 DVD obsahující text práce, implementovanou aplikaci a zdrojové kódy

## 1. Úvod

Investování a spekulace na finančním trhu je velice komplexní disciplína, která v sobě
zahrnuje principy ekonomie, matematiky, statistiky a informačních technologií. Proto i tato
bakalářská práce rozebírá toto téma více pohledy.

V teoretické části budou nejprve popsány vybrané trhy na finančním trhu a představeny jak
typy trhů, tak i trhy samotné. Následuje teorie o investičních strategiích. Z čeho se tyto
strategie skládají a jaké jsou přístupy při realizaci. Nakonec budou definovány tři vybrané
investiční strategie, u kterých bude přiblížen jak teoretický, tak matematický základ. Strategie
budou rozděleny na tržní a párové. Jakmile budou definovány trhy a investiční strategie, bude
následovat popis způsobů aplikací investičních strategií na trhy. Ze způsobů aplikace budou
představeny dva hlavní, ze kterých bude následně možno realizovat tzv. Out-of-sample test.
Popsány budou kroky, které jsou potřeba k provedení a co je cílem tohoto způsobu testování.
Poslední kapitolou v teoretické části bude popsáno, jak je možno za pomoci statistiky
vyhodnocovat výsledky dosažené po aplikaci investičních strategií na trhy. Díky této kapitole
bude v závěru možno porovnat dosažené výsledky.

V praktické části práce bude navrženo a naimplementováno řešení pro teoretickou část.
Nejprve budou sepsány požadavky na aplikaci, z nichž pak budou stanoveny případy užití. Po
analýze a návrhu následuje implementace, kde bude představeno jedno z možných řešení, jak
implementovat investiční strategie a způsob aplikace investiční strategie na vybrané trhy.
Jelikož budou potřeba data pro vybrané trhy, je nutno zajistit jejich stažení a představení
způsobu jejich stahování. Další součástí bude navržení a implementace komunikace se
statistickým softwarem. Důvodem komunikace je využívání statistických metod pro výpočty v
rámci investičních strategií a pro následnou analýzu výsledků. Nakonec budou popsány kroky
pro aplikaci investičních strategií na trhy pomocí software navrženého v bakalářské práci.

Cílem práce je přinést čtenáři ucelený pohled na teoretickou a praktickou stránku investování a
spekulace, čehož je docíleno vysvětlením a následnou implementací teoretických znalostí. V
rámci práce bude analyzována, navržena a následně implementována aplikace CSAnalyzer
umožňující aplikovat stanovené investiční strategie na vybrané trhy. Pomocí teoretických
základů a implementované aplikace bude v závěru možné analyzovat dosažené výsledky a
rozhodnout, zda-li je některý z dosažených výsledků vhodný pro reálné obchodování.

## 2. Teoretická část

V teoretické části práce si představíme trhy, na kterých budeme testovat investiční strategie.
Dozvíme se teoretický a matematický základ těchto investičních strategií spolu se způsobem,
jak tyto strategie aplikovat na vybrané trhy. Ze způsobů aplikace si představíme out-of-sample
způsob testování. Na závěr si ukážeme, jak pomocí statistiky analyzovat výsledky.

### 2.1 Trhy

Na finančním trhu je možné obchodovat velmi mnoho různých aktiv. Rozdělení těchto aktiv
ale nebudeme popisovat, namísto toho si popíšme teoretický základ ke všem aktivům. Trhy
jsou základní jednotkou v oblasti investování a spekulace, díky které jsme schopni realizovat
výnosy. Finanční instrumenty (neboli aktiva) mohou mít jednak podobu cenných papírů,
spořících účtů, finančních derivátů, pojišťovacích kontraktů, jednak reálných instrumentů
v podobě diamantů, drahých kovů, uměleckých sbírek, nerostných surovin a podobně.
Základní jednotkou těchto aktiv je vlastnost velikost ticku. **Tick** je nejmenší možná jednotka,
o kterou se cena aktiva hýbe. Některé trhy však mají velikost ticku odlišnou, proto je důležité
evidovat pro každé aktivum jeho velikost ticku. Dalším důležitým údajem je **minimální
obchodované množství** na daném aktivu. Díky těmto dvěma údajům jsme schopni spočítat,
kolik USD dostaneme v případě nakoupení minimálního množství aktiva při pohybu o jeden
tick, neboli **TickValue**. Trhy, které budeme v této práci analyzovat, jsme vybrali z **Nyse
Euronext**^1. Jedná se o euro americkou společnost, která provozuje více burz.

#### 2.1.1 ETF

ETF jsou otevřené fondy, které mají ve svém porfoliu určitý balík akcií. Čili investor či
obchodník, který má zájem sledovat svým portfoliem určitou skupinu či vybraný koš akcií, již
nemusí sám nakupovat všechny složky v odpovídajícím indexu do svého portfolia, nýbrž mu
stačí zobchodovat pouze akcii odpovídajícího ETF [23]. Tyto trhy jsme vybírali z Nyse Arca^2 ,
což je burza, která je součástí Nyse Euronext. Při výběru trhů jsme se zaměřili na hlavní
světové indexy. Vybrané trhy jsou **SPY** , což je ETF, jehož podkladovým aktivem je stejná
skupina aktiv jako je v indexu S&P 500. Dalším je **DIA** , který napodobuje svou skladbou aktiv
Dow-Jones index. Aktivum **IWM** napodobuje svou skladbou aktiv index Russell 2000.
1
2 Více https://globalderivatives.nyx.com/
Více https://globalderivatives.nyx.com/node/

Posledním vybraným aktivem je **EWG** , jež napodobuje MSCI neměcký index. Tabulka níže
pak zobrazuje důležité informace pro vybrané ETF trhy získané od společnosti IQFeed.

```
Tabulka 1: Vybraná ETF aktiva a jejich vlastnosti
```

```
Trh Popis
```

```
Velikost
ticku TickValue^
```

```
Min.
množství
DIA INDUSTRIAL AVER SPDR DOW JONES 0,01^1
EWG ISHARES MSCI GERMANY INDEX 0,01^1
IWM ISHARES RUSSELL 2000 INDEX 0,01^1
SPY SPDR S&P 500 0,01^1
Zdroj: IQFeed, Vlastní zpracování
```

#### 2.1.2 Futures

Futures kontrakt je dohodou dvou stran o uskutečnění obchodu v budoucnu. Obě strany
stanoví v den dohody všechna potřebná specifika obchodu, a jakmile nastane den vypořádání,
obchod se uskuteční za předem daných podmínek. Tyto futures kontrakty se obchodují na
komoditních burzách, kde spolu s opcemi tvoří aktiva vhodná k zajištění proti budoucnosti
[24]. Tyto trhy jsme vybírali z Nyse Liffe^3 , jež je také součástí Nyse Euronext. Při výběru
komoditních trhů jsme se zaměřili na dva hlavní světové trhy drahých kovů. Prvním vybraným
je trh zlata **ZG** , druhým vybraným pak trh stříbra **ZI**. Tyto trhy jsme vybrali, jelikož se jedná o
vysoce likvidní aktiva. Další výhodou je pak vysoká atraktivita futures kontraktů na drahé
kovy. Tabulka níže pak zobrazuje důležité informace pro vybrané Futures trhy, získané od
společnosti IQFeed.

```
Tabulka 2: Vybraná Futures aktiva a jejich vlastnosti
```

```
Trh Popis
```

```
Velikost
ticku
```

```
TickValue
```

```
Min.
množství
@ZI# GOLD 100 OZ APRIL 2012 0,^1 10
@ZG# SILVER 5000 OZ MAY 2012 0,001^5
Zdroj: IQFeed, Vlastní zpracování
```

3
Více https://globalderivatives.nyx.com/precious-metals/nyse-liffe-us

### 2.2 Investiční strategie

V této kapitole si představíme teoretický a matematický základ vybraných investičních
strategií.

Základním stavebním kamenem investičních strategií jsou fundamentální a technická analýza.
Tyto analýzy mohou být zkonstruovány ručně, nebo častěji, jsou zalgoritmizovány a používány
v některém z investičních softwarů jako indikátory. Prvním stavebním kamenem je
**fundamentální analýza** , jež zkoumá základní faktory ovlivňující vývoj ceny investičního
aktiva. Hlavními zkoumanými faktory jsou účetnictví firem, fundamentální data ekonomik,
počasí apod. Na základě vyhodnocení těchto faktorů pak obchodníci dělají svá investiční
rozhodnutí. Fundamentální analýza probíhá ve třech úrovních, první úrovní je zkoumání
makroekonomických ukazatelů. Mezi tyto ukazatele patří vývoj HDP dané země, státní
regulace apod. V další úrovni sledujeme fundamentální informace pro odvětví či sektor a v
poslední úrovni sledujeme fundamenty samotného podniku. Například jeho účetní uzávěrky,
změny ve vedení a investiční plány do budoucna [1]. Druhým stavebním kamenem je
**technická analýza**. Ta naopak vychází z předpokladu, že všechny faktory, které by mohli mít
vliv na cenu, jsou již v tržní ceně započítány. Podle technické analýzy není nebo je velmi malá,
prodleva u započítání nových informací do tržní ceny. Tím pádem pak není fluktuace tržní
ceny zapříčiněna jen fundamentálními informacemi, ale také například psychologií davu [1].
Čistě techničtí obchodníci považují za platnou teorii efektivních trhů [2], proto se podle této
teorie nevyplatí sledovat fundamentální data a ani se zajímat např. o vnitřní hodnotu aktiva.
Teorie efektivních trhů vznikla na konci 60. a začátkem 70. let Eugenem Famou [2] a
rozděluje nám trhy na slabě, středně a silně efektivní. Co vlastně znamená, že jsou trhy
efektivní? Použijeme překlad slov Eugena Famy [2, s.383], který říká: “ _Primární rolí kapitálových
trhů je alokace vlastnických práv na kapitálové statky v ekonomice. Obecně platí, že ideálem je trh, ve kterém
ceny poskytují přesné signály pro alokaci zdrojů: to jest, trh ve kterém firmy mohou dělat produkčně-investiční
rozhodnutí a kde investoři mohou volit mezi finančními instrumenty, které reprezentují vlastnická práva na
aktivity firem, za předpokladu, že ceny finančních instrumentů v každém čase 'plně reflektují' veškeré
dostupné informace. Trh, na kterém ceny vždy 'plně reflektují' dostupné informace, se nazývá 'efektivní'._ ”

Fundamentální i technická analýza mohou být konstuovány ručně nebo mohou být
zalgoritmizovány a používány jako tzv. indikátor. Indikátor je základním stavebním kamenem
při stavbě investičních strategií, proto je důležité rozlišovat úroveň algoritmizace daného
indikátoru. Pokud je důležité rozlišovat úroveň automatizace u indikátoru, bude zajisté

důležitá i úroveň automatizace celé investiční strategie, proto si představíme rozdělení
investičních strategií podle úrovně automatizace. V tomto směru máme tři možnosti, jak
investiční strategii aplikovat na aktiva. Prvním způsobem je **diskréční přístup** , kdy má
obchodník svá pravidla vstupu a výstupu do trhu daný, ale o samotnou exekuci příkazů a
hlídání pravidel se stará sám ručně. Problém ale nastává v případě, že budeme tuto strategii
chtít testovat na velkém množství historických dat, jelikož časová náročnost takovýchto testů
je obrovská. V případě, že bychom se rozhodli tuto strategii ještě optimalizovat^4 , bude časová
náročnost takových testů v řádu let. Tento přístup je vhodný pro spekulanty, kteří mají určitý
cit pro trh a nerozumí si tolik s informačními technologiemi (dále jen IT). Další využití
diskréčního přístupu bych viděl u strategií, které ze své podstaty nemohou být zalgoritmovány.

Pro obchodníky, kteří mají zkušenosti s IT, bych doporučoval **přístup mechanický**. V tomto
přístupu má obchodník všechna svá pravidla investiční strategie zalgoritmována a už se jen
stará o otevírání pozic a o administraci již otevřených pozic. Cílem tohoto přístupu je umožnit
obchodníkovi využívat v co největší míře IT, ale konečné rozhodnutí o vstupu do pozice,
výstupu z pozice apod. je ponecháno na obchodníkovi. Tento přístup je velice vhodný pro
širokou populaci, jelikož v sobě kloubí výhody manuální administrace pozic a výhody rychlosti
testování strategií pomocí IT.

Čistě **automatický přístup** má své výhody i nevýhody. Za obrovskou výhodu můžeme
považovat plnou podporu IT při tvorbě strategie i při jejím testování. V případě, že bychom se
rozhodli strategii otestovat na velkém množství trhů nebo se rozhodli zkoumat výsledky
optimalizace této strategie, jsme ušetření zdlouhavého ručního testování. Stačí nám jen
strategii zalgoritmovat v obchodním software, a pak ji aplikovat na vybrané trhy. Výběr trhů
pak záleží na preferencích a na výsledcích strategie. Nevýhody z čistě automatického přístupu
vyplývají z ponechání obchodování čistě na algoritmu strategie. Samozřejmě záleží na
sofistikovanosti daného algoritmu, ale všeobecně se dá říci, že nevýhodou je nemožnost
reakce algoritmu strategie na neznámé či nečekané situace na trhu.

#### 2.2.1 Tržní strategie

Za tržní strategie považujeme takové strategie, které aplikují pravidla fundamentální a
technické analýzy právě na jeden trh.

4
více 2.3.2 Optimalizace

**Buy and Hold**

Strategie Buy and Hold (neboli nakup a drž) je nejznámější a nejstarší strategií aplikovanou na
investiční aktiva. Jedná se o způsob, při kterém nakoupíme aktivum a po určitém čase ho
prodáme a realizujeme zisk či ztrátu. U této strategie spekulujeme vždy na růst daného aktiva,
proto je tato strategie velice nevhodná v obdobích, kdy ceny aktiv klesají. Z tohoto důvodu je
vhodné u této strategie sledovat výnos aktiva na historických datech a vstupovat pouze do
aktiv, které vykazují pozitivní očekávaný výnos.

**SMA crossover**

Základním stavebním kamenem této strategie je jednoduchý klouzavý průměr (neboli Simple
moving average). V technické analýze jsou známy i jiné druhy klouzavých průměrů než
jednoduchý, pro naše účely ale bude vyhovovat jenoduchý klouzavý průměr. Tento indikátor
se řadí do skupiny trendových indikátorů. Tyto indikátory se využívají hlavně pro identifikaci
různě dlouhých směrových pohybů trhu. To, jak dlouhý trend se snažíme těmito indikátory
zachytit, závisí hlavně na zvolené periodě indikátoru, ze které se pak počítá pomocí algoritmu
indikátoru jeho hodnota. Vzorec pro výpočet jednoduchého klouzavého průměru v čase _i_ je
zobrazen níže.

##### ᡅᠹᠧ 〒=^1 ᡦ 㔳 ᡶ〷

```
〒
```

```
〷⢀〒⡹ぁ⡸⡩
(1)
```

Za proměnnou _xj_ budeme dosazovat close ceny daného aktiva, v čase _j_ a _n_ nám značí periodu
tohoto indikátoru. Jednoduchý klouzavý průměr se řadí do základních indikátorů, jehož
principů se využívá nejenom v souvislosti s uzavírací cenou, jak si ukážeme později v
následující kapitole. Dalšími možnostmi, které se využívají u klouzavého průměru, jsou
způsoby manipulace s periodou a aktuální hodnotou. Lze totiž v tomto směru stanovit "okno"
náhledu, jež má velikost periody. Toto "okno" pak lze posunout od aktuální hodnoty směrem
do minulosti (jako je to v našem případě), nebo ho lze posunout do středu aktuální hodnoty a
"okno" je pak rozprostřeno okolo aktuální hodnoty. Matematické pojetí klouzavého průměru
lze interpretovat i graficky, jako na obrázku níže. Obrázek znázorňuje klouzavý průměr
počítaný z 50ti uzavíracích cen, pomocí aplikace CSAnalyzer, kterou popíšeme v kapitole 3.
Praktická část.

```
Obrázek 1: Grafické znázornění jednoduchého klouzavého průměru
```

```
Zdroj: [14], Vlastní zpracování
```

Tato strategie se řadí do **trendových strategií** neboli trend following strategií. Tyto strategie
se drží obecně známého rčení, že "trend je tvůj přítel". A proto se skládají především z
trendových indikátorů. Jsou to strategie, jejichž cílem je svézt se na základních růstech nebo
poklesech (trendech). Jejich procento úspěšnosti obchodů zpravidla nedosahuje ani 50%
(úspěšnost mezi 30-40 % je u těchto systemů naprosto běžná), protože strategie indikuje
spoustu signálů snažících se odhalit nový trend, nicméně pokud se později tento signál ukáže
jako falešný, je velmi brzy ukončen se ztrátou. Pokud se ale trend ukáže jako potvrzený,
strategie se snaží udržet se v trhu co nejdéle. To znamená až do doby, než trend ochabne.
Nevýhodou těchto strategií jsou však situace, kdy se trh pohybuje v kanálu (neboli range). V
těchto případech, při špatně zvolené periodě indikátorů, se strategie pokouší neustále
vstupovat do pozice, když už je trend vyčerpaný a vystupovat, když je trend proti otevřené
pozici. To má za následek neustálé střídání ztrátových pozic. Výsledky těchto strategií jsou
proto ovlivněny schopností aktiva tvořit silné trendy.

Základní pravidla strategie jsou generována na základě použití dvou klouzavých průměrů s
různými periodami. Jeden z klouzavých průměrů má zpravidla menší periodu než druhý, proto
se budou v průběhu času tyto klouzavé průměry křížit. A právě při překřížení vznikají signály
k nákupu a prodeji aktiva. **Do dlouhé pozice** budeme **vstupovat** , pokud klouzavý průměr s
menší periodou protne zespodu nahoru klouzavý průměr s větší periodou. Naopak **z dlouhé
pozice** budeme **vystupovat** , pokud klouzavý průměr s menší periodou protne shora dolů
klouzavý průměr s větší periodou. Pravidla pro **vstup a výstup do krátké pozice** jsou vice

versa pravidel vstupu a výstupu do dlouhé pozice, což zapříčiní, že tato strategie bude neustále
v obchodě, ať už na dlouhou či krátkou stranu [15].

```
Obrázek 2: Základní pravidla SMA crossover graficky
```

```
Zdroj: [14], Vlastní zpracování
```

Na předchozím obrázku je vstup do krátké pozice označen červeným trojúhelníkem
směřujícím vzhůru. Naopak výstup z krátké pozice je označen červeným trojúhelníkem
směřujícím dolů. Jelikož se pozice neustále střídají, výstup z jedné pozice znamená vstup do
opačné pozice (neboli position reverting). U výstupu z krátké pozice lze vidět i zelený
trojúhelník směřující vzhůru, který značí vstup do dlouhé pozice. Naopak u zeleného
trojúhelníku směřujícího dolů, který značí výstup z dlouhé pozice, lze vidět vstup do krátké
pozice. Barva čáry spojující vstup s výstupem je pak znázorněním ziskovosti obchodu, zelená
čára značí zisk z obchodu, červená čára pak značí ztrátu z obchodu.

#### 2.2.2 Párové strategie

Za párové strategie považujeme takové strategie, které aplikují pravidla fundamentální a
technické analýzy právě na dva trhy, při čemž nákup a prodej probíhá na obou aktivech
současně.

Důležitými pojmy jsou u těchto strategií tržní neutralita a závislost obou aktiv. **Tržní
neutralitu** je možné provést několika způsoby^5 , z nichž nejdůležitější je dolarová a kusová
neutrálnost [3, s.30]. Dolarová neutrálnost (neboli dollar neutrality) je pokud nakoupíme obě
5
dolarová,kusová,sektorová,beta,long/short

aktiva a do obou aktiv vložíme stejné finanční prostředky. Tento způsob neutrálnosti se
využívá především u statistické arbitráže^6 [3, s.29]. Naopak kusová neutrálnost (neboli share
neutrality) je pokud nakoupíme obě aktiva a od obou aktiv koupíme stejné množství [3, s.29].
Tento způsob neutrálnosti se využívá především u komoditních spreadů^7 a při hedgování^8 již
otevřených pozic. Neméně důležitým prvkem v párových strategiích je **závislost obou aktiv**.
Právě díky existenci závislosti mezi obchodovanými aktivy je možné realizovat výnosy s
nižším rizikem.

Způsobů, jak můžeme realizovat párovou strategii, je opravdu mnoho a podrobněji se tomuto
problému věnuje nespočet literárních děl [16, 17, 18]. Pro naše účely postačí, když si
představíme jednoduší způsob realizace párové strategie [4]. Prvním krokem je při výběru
aktiv do párů splnění podmínky závislosti obou aktiv, kterou budeme zjišťovat za pomocí
**korelačního koeficientu**. Ten nám udává sílu neboli těsnost závislosti dvou číselných
proměnných [5]. Vzorec pro výpočet korelačního koeficientu je zobrazen níže.

##### ᡰ げけ =ᡅᡅ けけげ ᡅ げ= ᡶᡷ − ᡶ.ᡷ

##### 㒕䙲 ᡶ⡰− ᡶ⡰ 䙳(ᡷ⡰− ᡷ⡰)

##### (2)

_Sxy_ značí kovarianci proměnných _X_ a _Y_ , _Sx_ značí směrodatnou odchylku proměnné _X_ a _Sy_
značí směrodatnou odchylku proměnné _Y_. Pokud budou obě aktiva vykazovat pozitivní
závislost a jejich korelační koeficient bude větší než 0.75, pak lze tuto závislost považovat za
dostatečně silnou a tento pár je pro nás vhodný k aplikaci párové investiční strategie.
Základním indikátorem pro námi zvolenou párovou strategii je poměr cen obou aktiv
vyvíjející se v čase (neboli price ratio indikátor). Vzorec je zobrazen níže.

##### ᡂᡄ 〒= ᡐᡑ 〒

```
〒
(3)
```

Price ratio indikátor indikátor vypočteme jako podíl close ceny aktiva _X_ v čase _i_ a aktiva _Y_ v
čase _i_ [4]. Dalším indikátorem bude pro naši strategii klouzavý průměr vypočítaný z price ratio
indikátoru.

6
7 Způsob párového obchodování využívaný k obchodování především akciových a ETF titulů^
8 Způsob párového obchodování využívaný k obchodování komoditních futures kontraktů
Způsob zajištění již otevřených pozic pomocí otevření další pozice na korelovaném aktivu

##### ᡅᠹᠧᡂᡄ 〒=^1 ᡦ 㔳 ᡂᡄ 〷

```
〒
```

```
〷⢀〒⡹ぁ⡸⡩
(4)
```

Vzorec pro tento výpočet se velice podobá vzorci 1.1 pro výpočet jednoduchého klouzavého
průměru. Rozdílem je však proměnná, kterou dosazujeme místo uzavírací ceny aktiva. V
případě vzorce 4.1 totiž dosazujeme hodnoty price ratio indikátoru v čase _j_ a za _n_ dosadíme
periodu tohoto indikátoru. Posledním důležitým indikátorem bude klouzavý průměr
vypočítaný z price ratio indikátoru, ale k tomuto indikátoru bude v jednom případě přičtena, v
druhém případě odečtena, hodnota **dvou směrodatných odchylek σ**. Vzorec pro výpočet
směrodatné odchylky z price ratio indikátoru je zobrazen níže.

##### ᡅᡂᡄ 〒= 㒗 ᡦ − 1^1 㔳 (ᡂᡄ 〷− ᡅᠹᠧᡂᡄ 〒)⡰

```
〒
```

```
〷⢀〒⡹ぁ⡸⡩
```

```
(5)
```

_PRj_ značí hodnotu price ratio indikátoru v čase _j_ , _SMAPR i_ pak značí průměrnou hodnotu
tohoto indikátoru za periodu _n_. Pokud sečteme rozdíly _PR j_ od této průměrné hodnoty,
umocníme je a vydělíme _n-1_ , dostaneme hodnotu, kterou v posledním kroku odmocníme.
Jedná se tedy o zprůměrování druhých mocnin rozdílů mezi _PR j_ a _SMAPR i_. Tímto
způsobem jsme schopni spočítat velikost směrodatné odchylky, ale potřebujeme ještě vyjádřit
přičtení a odečtení této hodnoty od klouzavého průměru price ratio indikátoru. Tento vztah je
zobrazen na vzorci níže.

```
ᡅᠹᠧᡂᡄᡅ 〒=	ᡅᠹᠧᡂᡄ 〒±	ᡅᡂᡄ 〒
```

```
(6)
```

_SMAPR i_ značí klouzavý průměr z price ratio indikátoru v čase _i_ , ke kterému
přičteme/odečteme hodnotu _SPR i_ (neboli hodnotu směrodatné odchylky price ratio
indikátoru) [4].

U strategie budeme tedy používat čtyři indikátory, tři klouzavé průměry a jeden price ratio
indikátor. V grafickém znázornění bude price ratio indikátor oscilovat okolo klouzavého
průměru a občas překročí jeden z průměrů posunutých o ±2σ, tyto situace budeme
vyhledávat. Při těchto situacích nám strategie říká, že poměr mezi dvěma aktivy překročil

průměrnou hodnotu za posledních _n_ barů o více než dvojnásobek průměrné odchylky σ. V
těchto situacích budeme předpokládat, že by se hodnota price ratio indikátoru měla vrátit opět
k průměrné hodnotě, čili k hodnotě klouzavého průměru. Tento přístup, kdy předpokládáme
návrat k průměrné hodnotě, se nazývá **mean-reversion**.

```
Obrázek 3: Grafické znázornění indikárů pro párovou strategii
```

```
Zdroj: [14], Vlastní zpracování
```

První dva horní grafy zobrazují vývoj obou aktiv, spodní graf pak zobrazuje všechny čtyři
indikátory. Všechny grafy mají sjednocené osy _X_ , proto si lze všimnout, že obě aktiva se
opravdu pohybují v určité závislosti. Červená linka reprezentuje price ratio indikátor, světle
modrá linka je klouzavý průměr a zbylé dvě linky jsou klouzavé průměry posunuté o ±2σ. Jak
lze vidět na spodním grafu v Obrázku 3, podíl cen aktiv (neboli price ratio indikátor) se
pohybuje v úzkém rozmezí a jedná se o malé pohyby, které jsou často špatně graficky
interpretovatelné. Proto lze všechny tyto čtyři indikátory shrnout do jednoho jediného
indikátoru, díky kterému se nám pravidla této strategie zpřehlední a zlepší svou vypovídací
schopnost. Vzorec výpočtu normalizovaného zobrazení je zobrazen níže.

```
ᡦᡧᡰᡥ 〒=	ᡂᡄ 〒−	ᡅᠹᠧᡂᡄᡅᡂᡄ 〒
〒
(7)
```

Normalizací rozumíme odečtení hodnoty price ratio indikátoru v čase _i_ ( _PR i_ ) od klouzavého
průměru price ratio indikátoru v čase _i_ ( _SMAPR i_ ), děleného směrodatnou odchylkou z price
ratio indikátoru v čase _i_ ( _SPR i_ ). Díky tomuto kroku jsme docílili normalizovaného zobrazení
indikátorů párové strategie.

```
Obrázek 4: Grafické znázornění indikárů pro párovou strategii v normalizovaném tvaru
```

```
Zdroj: [14], Vlastní zpracování
```

Jak je vidět na obrázku výše, docílili jsme spojení čtyř indikátorů do jednoho. Tento indikátor
nám značí vývoj price ratio indikátoru v čase _i_ ( _PRi_ ) v relativní formě (normalizované formě).
Indikátor klouzavého průměru price ratio indikátoru v čase _i_ ( _SMAPRi_ ) je reprezentován
nulovou linkou. Pokud tedy normalizovaný indikátor překročí nulovou linku, jedná se o
stejnou situaci, jako když _PRi_ překročí indikátor _SMAPRi_. Indikátory klouzavých průměrů
posunutých o ±2σ ( _SMAPRSi_ ) jsou reprezentovány linkami o hodnotách ±2. Pokud tedy
normalizovaný indikátor překročí jednu z linek -2 či 2, jedná se o stejnou situaci, jako když
price ratio indikátor ( _PRi_ ) překročí jednu z linek _SMAPRSi_. Díky normalizaci můžeme i
snadněji interpretovat graficky základní pravidla párové strategie, jelikož osa _X_ v tomto
indikátoru osciluje okolo nulové linky a v závislosti na velikosti odchylky se pohybuje v
určitém rozmězí. Nejčastěji pak v rozmezí (-3,3). Pravidla párové strategie máme zobrazeny na
obrázku níže.

```
Obrázek 5: Základní pravidla párové strategie graficky
```

```
Zdroj: [14], Vlastní zpracování
```

Základní pravidla strategie se pak řídí pouze podle červené linky ve spodním grafu. Jestliže
linka překročí hodnotu 2, jedná se o **signál k prodeji celého páru**. Pod tímto pojmem si
představme prodej prvního aktiva a nákup druhého aktiva. Tato strategie je dolarově neutrální,
takže budeme nakupovat a prodávat vždy stejně velký objem investovaných prostředků. V
případě, že se linka vrátí pod hodnotu 0, jedná se o **signál k ukončení prodeje celého páru**.
V tomto případě uzavřeme obě pozice na aktivech. Opačná situace nastane v případě, že
červená linka klesne pod -2. V tuto chvíli se jedná o **signál k nákupu celého páru** , což v
dílčích úkonech znamená nákup prvního aktiva v páru a prodej druhého aktiva v páru.
V případě, že linka stoupne nad 0, jedná se o signál k ukončení nákupu celého páru, ukončíme
tedy pozice na obou aktivech [4].

### 2.3 Aplikace investiční strategie

V této kapitole představíme způsob aplikace investiční strategie na trhy. Pro tuto kapitolu je
důležité uvědomit si význam informací uvedených v předchozích kapitolách. Jak vyplývá z
popisu indikátorů pro uvedené investiční strategie, většina z nich potřebuje pro svůj výpočet
alespoň jeden **vstupní parametr**. Strategie se však skládá z jednotlivých indikátorů, a proto

bude mít i strategie vstupní parametry. Díky těmto parametrům je strategie schopna
inicializovat své indikátory pro plnění svých pravidel. V závislosti na složitosti strategie, neboli
v závislosti na počtu použitých indikátorů, se pak zvyšuje či snižuje počet potřebných
vstupních parametrů.

#### 2.3.1 Backtest

Historická simulace (neboli backtest) je způsob, jak zjistit výkonnost investiční strategie na
historických datech vybraného trhu. Výkonnost na historických datech je pro nás klíčovou
informací, jelikož lze předpokládat, že **strategie nebude zisková, pokud nebyla zisková na
historických datech**. Z toho důvodu je nutné evidovat obchody, které strategie realizovala
při historické simulaci a z nich pak sestrojit statistiky popisující výkonnost strategie na
vybraném trhu [6, s.93]. Popis statistik bude rozebrán v kapitole 2.4 Analýza výsledků.

#### 2.3.2 Optimalizace

Při historické simulaci získáme informace o výkonnosti jedné strategie s nastavenými
parametry aplikované na jednom trhu. Pokud ale budeme chtít mít představu například o
výkonnosti jedné strategie s různými parametry aplikované na jednom trhu, budeme používat
optimalizaci. Výhodou optimalizace je, že **získáme představu o výkonnosti strategie napříč
zadanými parametry** [6, s.211]. Díky tomuto přístupu můžeme kontrolovat tzv. robusnost
strategie. Jednoduše lze říci, že pokud strategie vykazuje pozitivní očekávaný výnos u většiny
zadaných parametrů, jedná se o velice robusní strategii. A výsledky této strategie by neměly být
ovlivňeny drobnými změnami v chování trhu. Další kritérium při optimalizaci bude výběrové
kritérium, podle kterého si můžeme výsledky seřadit a vybrat takové nastavení parametrů,
které bude vyhovující.

Způsobů realizace optimalizace je mnoho, pro naše účely jsme zvolili jednoduchý přístup za
pomocí tzv. Simple optimization [6, s.214]. Tento způsob definuje parametry strategie jako
sekvenci čísel, nikoliv pevně stanovená čísla. A historická simulace se provede pro každou
hodnotu z této sekvence. Jestliže pak budeme porovnávat jednu statistiku z výsledků mezi
různými vstupními parametry strategie, lze je porovnávat i graficky. Lze tedy kontrolovat i
robusnost strategie graficky. Grafické vyjádření má však smysl pouze u jednoho či dvou
vstupních parametrů, jelikož maximální prostor, který umíme výjádřit, je trojrozměrný. Z
tohoto důvodu budeme používat především 2D a 3D grafy.

```
Obrázek 6: Grafické znázornění optimalizace dvou vstupních parametrů
```

```
Zdroj: http://computeraidedfinance.com/2012/01/03/how-can-i-optimize-my-quantitative-trading-strategy-e-g-
macd-signal/
```

Obrázek nám graficky znázorňuje velikost kumulovaného výnosu (osa _Z_ ) investiční strategie v
závislosti na dvou vstupních parametrech (osy _X_ , _Y_ ). Díky tomuto zobrazení jsme schopni
rychle identifikovat robusní strategii, jelikož svoji pozornost zaměřujeme, v tomto případě,
hlavně na světlé oblasti. V těchto oblastech totiž dosahovala strategie nejlepších výsledků.
Celkového nejlepšího výsledku (neboli globálního maxima) bylo dosaženo s hodnotami
vstupních parametrů 13 a 31, s nimiž strategie dosáhla kumulovaného výnosu 6,276.

#### 2.3.3 Out-of-sample test

Out-of-sample testování (neboli Walk-Forward analýza) je způsob, jak využít optimalizace a
backtestu k získání dalších informací o výkonnosti strategie. Tímto způsobem testování
**ověříme, že výsledky dosažené v optimalizační fázi, jsme schopni replikovat na data,
která v optimalizační fázi nebyla zahrnuta**. Díky tomuto postupu ověříme robusnost a
případně vyřadíme přefitované kombinace trhu a strategie.

Jedná se o dvoukrokové testování, kdy si data trhu rozdělíme v poměru odpovídající zhruba
75% a 25%. Větší část dat bude sloužit na **in-sample optimalizaci** (IS)^9. V této fázi proběhne

9
Viz 2.3.2 Optimalizace

optimalizace parametrů strategie a dojde k výběru nejlepších parametrů s ohledem k
hledanému kritériu.
Obrázek 7: In-sample optimalizace

```
Zdroj: [14], Vlastní zpracování
```

Menší část dat bude sloužit na **out-of-sample backtest** (OOS). V této fázi se provede
historická simulace na datech s danými parametry strategie, které jsme získali z in-sample
optimalizace.

```
Obrázek 8: Out-of-sample backtest
```

```
Zdroj: [14], Vlastní zpracování
```

Tento postup je důležitý z hlediska věrohodnosti výsledků. Pokud totiž bude strategie
vykazovat pozitivní očekávání při optimalizaci, ale už nikoliv při historické simulaci na out-of-
sample datech, není možné takové strategii důvěřovat, že bude výsledky reflektovat i při
reálném obchodování. Z výše uvedených dvou kroků také dostaneme dvě různé statistiky
výkonnosti strategie. Tyto dvě statistiky mezi sebou chceme porovnat, abychom zjistili, zdali si
strategie udržela výkonost z optimalizačního procesu i na datech v out-of-sample testu. Tento
krok je klíčový k identifikaci přeoptimalizovaných výsledků [6, s.239].

**Přeoptimalizace** je indikována v případě, že si strategie neudrží svoji výkonnost z
optimalizační fáze v out-of-sample fázi. Konečným potvrzením přeoptimalizace je pak při
přechodu na reálné obchodování, kdy tyto strategie nedosahují ani zdaleka výkonnosti v
optimalizační fázi. V našem případě budeme indikovat přeoptimalizaci pomocí Walk-forward
Efficiency (dále jen WFE). Podle tohoto způsobu je přeoptimalizace identifikována, pokud
anualizovaný výnos strategie v out-of-sample testu nedosáhne alespoň 60% anualizovaného
výnosu z in-sample optimalizace [6, s.239].

### 2.4 Analýza výsledků

V této kapitole si představíme většinu ukazatelů, které budeme po aplikaci investiční strategie
vyhodnocovat. Jelikož v praktické části budeme popisovat softwarové řešení na hodnocení
výsledků investičních strategií, bylo by správné říci si ještě před praktickou částí teoretickou a
matematickou rovinu pro tyto ukazatele. Základní jednotkou pro výpočty budou realizované
výnosy z obchodu. Nicméně stejné ukazatele lze spočítat pro jakékoliv jiné výnosy, například
pro výnosy z aktiva či výnosy strategie z hlediska pozice.

**Výnos** můžeme vyjádřit dvěma způsoby. V **absolutní míře** znamená, že vyjadřujeme výnos v
absolutních jednotkách, to jest vyjádření v tickách nebo v dolarech. Tick je nejmenší možná
jednotka, o kterou se cena aktiva pohne. Kdybychom vyjadřovali výnos pouze v těchto
jednotkách a chtěli mezi sebou porovnávat výsledky strategie aplikované na více trhů, které
budou investovat vždy do každého obchodu například 100 kusů (neboli tato strategie bude
kusově neutrální z pohledu realizovaných obchodů), nepodařilo by se nám dosáhnout
správného porovnání. Důvodem by v tomto případě byla rozdílná cena aktiv. Výnos v
**relativní míře** je v tomto ohledu lepším způsobem. Jelikož se jedná o vyjádření relativní míry
pomocí procentuálního počtu, lze předpokládat správnou porovnatelnost dosažených
výsledků. Prvním způsobem je výpočet aritmetického výnosu, který je zobrazen níže.

##### ᡄ 〨ぅ〒ぇ =ᡐ ぇ ⡸⡩ᡐ− ᡐ ぇ

```
ぇ
(8)
```

U tohto vzorce záleží na tom, zdali chceme počítat výnos aktiva, či výnos strategie podle
provedených obchodů. _Xt_ nám značí uzavírací cenu aktiva v čase _t_ (nebo vstupní cenu u
obchodu) a _Xt+1_ nám značí uzavírací cenu v čase _t+1_ (nebo výstupní cenu u obchodu).

Druhým způsobem je logaritmický výnos, který je pro použití vhodnější než průměr
aritmetický, jelikož může nabývat hodnot z intervalu (-∞,∞). V tomto případě vidíme, že
interval je symetrický, to je vhodné zejména pro případ, kdy budeme uvažovat normální
rozdělení výnosů. Naopak aritmetický průměr může nabývat pouze hodnot z intervalu (-1,∞) a
to je nevhodné pro uvažování normálního rozdělení výnosů, jelikož interval není symetrický.
Výpočet pro logaritmický výnos je uveden na následujícím vzorci.

```
ᡄ卄あ〴 = log	(	ᡐᡐぇ⡸⡩
ぇ
```

##### )

##### (9)

U tohoto vzorce opět záleží na tom, zdali chceme počítat výnos aktiva či výnos strategie podle
provedených obchodů. _Xt_ nám značí uzavírací cenu aktiva v čase _t_ (nebo vstupní cenu u
obchodu) a _Xt+1_ nám značí uzavírací cenu v čase _t+1_ (nebo výstupní cenu u obchodu).

Tím, že jsme si stanovili relativní míru výnosu, jsme sjednotili výnosy napříč všemi strategiemi
a trhy z hlediska porovnání. Toto porovnání je ale pouze z hlediska výnosu (neboli osy _X_ ). Co
když ale nastane situace, kdy budeme chtít porovnávat výnosnost trhů, které se liší svou
délkou dat? U prvního trhu známe například uzavírací ceny rok zpět, u druhého například
deset let zpět. V tomto případě využijeme **anualizovaný výnos** , který nám pomůže sjednotit
dosažené výnosy z hlediska času. Výpočet anualizovaného výnosu zobrazuje následujíc vzorec.

```
ᠧᡦᡦᡳᡓᡤ	ᡄ = 㐶	ᡄ〰〲卄十ᡖ 	㑀.365
```

```
(10)
```

_Rcelk_ značí celkový výnos aktiva či strategie, _d_ značí počet dní, ze kterých jsme tento výnos
počítali.

#### 2.4.1 Popisná statistika

V této části představíme možnosti analýzy výnosů pomocí popisné statistiky. Stejně jako u
každého jiného statistického souboru můžeme počítat statistiky pro soubor výnosů. Za
**historický výnos** budeme považovat aritmetický průměr (neboli střední hodnotu) z výnosů
[10, s.14]. K historickému výnosu můžeme evidovat také dva krajní kvantily, které nám
pomohou lépe popsat rozložení veličin kolem střední hodnoty. Tyto hodnoty se vztahují k
historickému výnosu, což je pouze jedna strana mince. Investor by se měl ale soustředit na
více veličin než jen na výnos, což je naznačeno na obrázku magického trojúhelníku investora
[11, s.24].

```
Obrázek 9: Magický trojúhelník investora
```

```
Zdroj: [11, s.24, obr.1.2]
```

**Likviditu** budeme ignorovat, jelikož předpokládáme dostatečnou likviditu na vybraných
trzích. Další veličinou, kterou ale ignorovat nebudeme, je riziko. **Riziko** budeme měřit podle
směrodatné odchylky, rozptylu a variačního koeficientu [10, s.15], z nichž je poslední
jmenovaný vyjádřením relativním a tedy porovnatelným napříč trhy a strategiemi.

Další ukazatel, který budeme sledovat, je **kumulovaný výnos** (neboli equity) vyvíjející se v
čase. Ten získáme jako úhrn všech výnosů do okamžiku _i_ podle následujícího vzorce [6, s.104].

##### ᡕᡳᡥ ᡄ 〒= 㔳 ᡶ〷

```
〒
```

```
〷⢀⡩
(11)
```

Z křivky kumulovaného výnosu ale můžeme získat i další zajímavé údaje. Jedním z těch, které
zastupují riziko, je drawdown [6, s.83]. Výpočet drawdownu je netriviální záležitost, proto je
nebudeme vyjadřovat vzorcem, ale popíšeme ji slovně. Peak je stav, kdy dosáhneme na křivce
kumulovaného výnosu nového maxima. Jakmile se nenacházíme ve stavu Peak, nacházíme se
v **drawdownu**. Pokud budeme drawdown evidovat jako řadu hodnot vyvíjející se v čase,
můžeme pro toto pole počítat průměrný a maximální drawdown, stejně jako průměrnou a
maximální dobu trvání drawdownu.

Křivku kumulovaného výnosu můžeme analyzovat i pomocí **lineární regrese** [12, s.180],
která nám pomůže měřit mimo jiné hladkost křivky z pohledu času. Lineární regrese popisuje
vztah dvou proměnných, z nichž jedna je brána jako závislá (neboli vysvětlovaná) a druhá jako
nezávislá (neboli vysvětlující). Proměnnou závislou značíme _X_ a proměnnou nezávislou _Y_.
Vhodné parametry regresní přímky pak odhadujeme pomocí metody nejmenších čtverců,
která je založena na minimalizaci residuálního součtu čtverců. Obecný vzorec regresní přímky
je stejný jako obecný vzorec lineární funkce.

```
ᡘ(ᡶ) = ᡓ + ᡔᡶ
(12)
```

Parametr _b_ pak nazýváme regresním koeficientem, který nám udává, o kolik se změní závislá
proměnná, když se nezávislá proměnná změní o jednotku [13]. Měření hladkosti equity křivky
se provádí za pomocí **Standard Errors** (SE) v lineární regresi. Čím větší SE, tím hrubější bude
equity křivka. Jinými slovy hledáme takovou equity křivku, která bude mít SE co nejmenší.
Dalším ukazatelem je **koeficient determinace** (neboli R2), který vyjadřuje těsnosti závislosti.
Posledním ukazatelem je **RRR** (neboli risk reward ratio), které spočítáme jako regresní
koeficient dělený SE. Jedná se o ukazatel, který poměřuje výnos proti riziku, kdy je pro nás
výnosem regresní koeficient a rizikem je pro nás SE [12, s.185].

**Grafy** , které se nejčastěji používají v popisné statistice, jsou line graf, box plot graf a bar chart.
Všechny tyto typy grafů jsou popsány například v dokumentaci k software Wolfram
Mathematica 8.0.

#### 2.4.2 Teorie pravděpodobnosti

V této části si ukážeme, jak lze aplikovat teorii pravděpodobnosti na soubor výnosů. U
souboru jsme již spočítali historický výnos, ale chtěli bychom také znát **očekávaný výnos**^10.
Očekávanou střední hodnotu lze spočítat jako vážený aritmetický průměr, kdy jako váhy
bereme v úvahu pravděpodobnosti jednotlivých náhodných veličin [10, s.7]. Jelikož jsou ale
výnosy spojitá náhodná veličina, nikoliv však diskrétní náhodná veličina, lze tento postup
aplikovat pouze pokud výnosy rozdělíme do několika intervalů. Z těchto intervalů jsme pak
schopni sestavit tabulku relativních četností. V tomto případě ale budeme považovat za
náhodnou veličinu ve vzorci tento interval, čímž docílíme větší nepřesnosti z hlediska
konečného očekávaného výnosu. Z tohoto důvodu jsme se rozhodli za očekávaný výnos
považovat námi spočítaný historický výnos. Ten nám podává lepší a přesnější informaci o
možném očekávaném výnosu.

Teorie pravděpodobnosti se opírá hlavně o pravděpodobnostní rozdělení náhodných veličin,
ze kterých jsou pak možné vypočíst další veličiny. My jsme zvolili normální rozdělení jako
pravděpodobnostní rozdělení výnosů. První veličinou, která nás bude zajímat, je
pravděpodobnost toho, že následující výnos bude větší než nula. To nám zobrazuje obrázek
níže.

```
Obrázek 10: Pravděpodobnost kladného výnosu
```

```
Zdroj: Wolfram Mathematica, Vlastní zpracování
```

10
více [http://www.investopedia.com/terms/e/expectedreturn.asp](http://www.investopedia.com/terms/e/expectedreturn.asp)

Na obrázku výše vidíme distribuční funkci normálního rozdělení. Pokud bychom chtěli zjistit
pravděpodobnost, že následující hodnota bude menší nebo rovna než 0 (neboli
ᡂ(ᡶ ≤ 0)), stačí zjistit hodnotu distribuční funkce v bodě 0. V tomto případě bychom dostali
hodnotu cca 0.4 (neboli 40 %). Pokud ale budeme chtít zjistit pravděpodobnost, že následující
hodnota bude větší než 0 (neboli ᡂ(ᡶ > 0)), musíme odečíst hodnotu distribuční funkce v
bodě 0 od hodnoty 1 (neboli 1 − ᡂ(ᡶ ≤ 0)). Což je v našem případě cca 0.6 (neboli 60 %).

Další veličinou, kterou můžeme spočítat z rozdělení pravděpodobnosti, je Value-at-Risk (dále
jen VaRisk). Jedná se o způsob měření rizika z pohledu pravděpodobnosti. My jsme pro náš
účel zvolili velice jednoduchý výpočet VaRisk, který jsme získali výpočtem kvantilu
normálního rozdělení v 10 % celkového rozsahu. Opačným způsobem lze spočítat i naší
vlastní metriku, Value-at-Return (dále jen VaReturn), která je opakem VaRisk. Obě veličiny
jsou zobrazeny na obrázku níže.

```
Obrázek 11: Hustota pravděpodobnosti s VaRisk a VaReturn
```

```
Zdroj: Wolfram Mathematica, Vlastní zpracování
```

Na obrázku výše vidíme histogram spolu s hustotou pravděpodobnosti normálního rozdělení.
Střední hodnota je na obrázku zobrazena žlutou svislou čárou. VaRisk je zobrazen jako
červená výplň v hustotě pravděpodobnosti, jedná se tedy o část hustoty, kdy je výnos menší
než 10% kvantil. Tuto hodnotu pak můžeme považovat za podstoupené riziko, jelikož v deseti
procentech případů pak můžeme očekávat výnos menší než je tato hodnota. Opakem je pak
VaReturn, který je zobrazen jako zelená výplň části hustoty pravděpodobnosti. Tato hodnota
nám pak říká, že v 10 % případů můžeme očekávat výnos větší než je hodnota VaReturn.

Posledními veličinami jsou pro nás testy normality. Těmito testy zjišťujeme, zdali námi
zvolené normální rozdělení opravdu patří na soubor výnosů. Neboli zjišťujeme, zdali má
soubor výnosů opravdu normální rozdělení, které my předpokládáme. Jako testy normality
jsme zvolili Kolmogorov-Smirnov test a Shapiro-Wilk test [19].

## 3. Praktická část

V této kapitole navrhneme a naimplementujeme rešení pro teoretickou část bakalářské práce.
Cílem je představit jedno z možných řešení, jak implementovat investiční strategie a způsob
aplikace investiční strategie na vybrané trhy. Jelikož budeme potřebovat data pro vybrané trhy,
musíme zajistit stažení těchto dat a představit způsob stahování. Další nutnou součástí bude
navržení a implementace komunikace se statistickým softwarem. Důvodem komunikace je
využívání statistických metod pro výpočty v rámci investičních strategií a pro následnou
analýzu výsledků.

### 3.1 Analýza a návrh

V této kapitole stanovíme funkční a nefunkční požadavky na aplikaci. Z identifikovaných
požadavků vytvoříme seznam případů užití a nakonec představíme high-level pohled na
navrženou architekturu aplikace.

#### 3.1.1 Funkční požadavky

- Aplikace umožní uživateli stahovat ETF a Futures data od poskytovatele dat IQFeed
- Aplikace umožní uživateli nahlížet na ETF a Futures data stažená z IQFeed
- Aplikace umožní uživateli vytvářet backtest investiční analýzy na vybraný trh
- Aplikace umožní uživateli vytvářet backtest investiční strategie na vybraný trh
  o Aplikace umožní uživateli ukládat pouze backtesty vyhovující zadaným
  podmínkám
- Aplikace umožní uživateli vytvářet optimalizaci investiční strategie na vybraný trh
  o Aplikace umožní uživateli ukládat pouze optimalizace vyhovující zadaným
  podmínkám
- Aplikace umožní uživateli nahlížet na provedené backtesty investiční analýzy
- Aplikace umožní uživateli nahlížet na provedené backtesty investiční strategie
  o Aplikace umožní uživateli filtrovat načtené výsledky podle zadaných podmínek
- Aplikace umožní uživateli nahlížet na provedené optimalizace investiční strategie
  o Aplikace umožní uživateli filtrovat načtené výsledky podle zadaných podmínek

#### 3.1.2 Nefunkční požadavky

- Aplikace bude jednoduše rozšiřitelná
- Aplikace bude mít charakter 3 vrstvé architektury
- Aplikace bude mít intuitivní ovládání
- Aplikace bude jednotlivé backtesty a optimalizace serializovat do souboru
- Aplikace bude komunikovat se statistickým softwarem Wolfram Mathematica 8.0
- Aplikace bude obsahovat tržní analýzy
  o SMA
- Aplikace bude obsahovat tržní strategie
  o SMAcrossover
- Aplikace bude obsahovat párové analýzy
  o Ratio
  o RatioMean
  o RatioMeanWithStdDeviation
  o RatioPlusMinus
  o RatioStdDev
- Aplikace bude obsahovat párové strategie
  o Ratio_2d_0
- Aplikace bude obsahovat analýzu výsledků pro historické simulace investičních
  strategií

#### 3.1.3 Seznam a popis případů užití

Seznam případů užití jsme vytvořili ze zadaných požadavků na aplikaci. Případy užití se vytváří
pro funkční požadavky, ale při vytváření seznamu případů užití jsme museli brát v potaz i
rozdělení investičních analýz, strategií a způsobů aplikace na tržní a párové. Identifikované
případy užití znázorňuje tabulka níže.

```
Tabulka 1: Seznam a popis případů užití
```

**Identifikátor Název případu užití**

**UC01** Stažení dat

**UC02** Zobrazení dat

**UC03** Spuštění backtestu tržní investiční analýzy

**UC04** Zobrazení backtestu tržní investiční analýzy

**UC05** Spuštění backtestu tržní investiční strategie

**UC06** Zobrazení backtestu tržní investiční strategie

**UC07** Spuštění optimalizace tržní investiční strategie

**UC08** Zobrazení optimalizace tržní investiční strategie

**UC09** Spuštění backtestu párové investiční analýzy

**UC10** Zobrazení backtestu párové investiční analýzy

**UC11** Spuštění backtestu párové investiční strategie

**UC12** Zobrazení backtestu párové investiční strategie

**UC13** Spuštění optimalizace párové investiční strategie

**UC14** Zobrazení optimalizace párové investiční strategie

```
Zdroj: Vlastní zpracování
```

**UC01 – Stažení dat**

1. Aplikace zobrazí okno pro zadání následujících informací
   a. Tickery stahovaných dat
   b. Datum od
   c. Datum do
   d. Typ aktiva
   e. TF aktiva
2. Uživatel načte zadané tickery z číselníku, z kterého se doplní některé informace o
   aktivech
3. Uživatel potvrdí zadané informace stiskem tlačítka Stáhni z IQFeed do...
4. Uživatel napíše jméno souboru, do kterého se budou data serializovat
5. Aplikace stáhne a serializuje zadaná data

**UC02 – Zobrazení dat**

1. Aplikace zobrazí okno pro načtení stažených dat
2. Uživatel vybere, ze kterého souboru chce načíst stažená data
3. Aplikace deserializuje a načte data do tabulky

**UC03 – Spuštění backtestu tržní investiční analýzy**

1. Aplikace zobrazí okno pro zadání následujících informací
   a. Název souboru se serializovanými daty
   b. Tickery dat
   c. Typ aktiva
   d. TF aktiva
   e. Název analýzy včetně jejích parametrů
2. Uživatel přidá zadané informace do exekučního plánu
3. Uživatel potvrdí spuštění historické simulace podle exekučního plánu
4. Aplikace spustí simulace a výsledné simulace bude serializovat do zadaného souboru

**UC04 – Zobrazení backtestu tržní investiční analýzy**

1. Aplikace zobrazí okno pro načtení tržních investičních analýz
2. Uživatel vybere, ze kterého souboru chce načíst backtesty
3. Aplikace deserializuje a načte data do tabulky

**UC05 – Spuštění backtestu tržní investiční strategie**

1. Aplikace zobrazí okno pro zadání následujících informací
   a. Název souboru se serializovanými daty
   b. Tickery dat
   c. Typ aktiva
   d. TF aktiva
   e. Název strategie včetně jejích parametrů
2. Uživatel přidá zadané informace do exekučního plánu
3. Uživatel potvrdí spuštění historické simulace podle exekučního plánu
4. Aplikace spustí simulace a výsledné simulace bude serializovat do zadaného souboru

**UC06 – Zobrazení backtestu tržní investiční strategie**

1. Aplikace zobrazí okno pro načtení tržních investičních strategií
2. Uživatel vybere, ze kterého souboru chce načíst backtesty
3. Aplikace deserializuje a načte data do tabulky

**UC07 – Spuštění optimalizace tržní investiční strategie**

1. Aplikace zobrazí okno pro zadání následujících informací
   a. Název souboru se serializovanými daty
   b. Tickery dat
   c. Typ aktiva
   d. TF aktiva
   e. Název strategie včetně rozsahu parametrů
2. Uživatel přidá zadané informace do exekučního plánu
3. Uživatel potvrdí spuštění optimalizací podle exekučního plánu
4. Aplikace spustí optimalizace a výsledky bude serializovat do zadaného souboru

**UC08 – Zobrazení optimalizace tržní investiční strategie**

1. Aplikace zobrazí okno pro načtení tržních investičních strategií
2. Uživatel vybere, ze kterého souboru chce načíst optimalizace
3. Aplikace deserializuje a načte data do tabulky

**UC09 – Spuštění backtestu párové investiční analýzy**

1. Aplikace zobrazí okno pro zadání následujících informací
   a. Název souboru se serializovanými daty
   b. Tickery dat
   c. Typ aktiva
   d. TF aktiva
   e. Název analýzy včetně jejích parametrů
2. Uživatel přidá zadané informace do exekučního plánu

3. Uživatel potvrdí spuštění backtestu podle exekučního plánu
4. Aplikace spustí simulace a výsledné simulace bude serializovat do zadaného souboru

**UC10 – Zobrazení backtestu párové investiční analýzy**

1. Aplikace zobrazí okno pro načtení párových investičních analýz
2. Uživatel vybere, ze kterého souboru chce načíst backtesty
3. Aplikace deserializuje a načte data do tabulky

**UC11 – Spuštění backtestu párové investiční strategie**

1. Aplikace zobrazí okno pro zadání následujících informací
   a. Název souboru se serializovanými daty
   b. Tickery dat
   c. Typ aktiva
   d. TF aktiva
   e. Název strategie včetně jejích parametrů
2. Uživatel přidá zadané informace do exekučního plánu
3. Uživatel potvrdí spuštění historické simulace podle exekučního plánu
4. Aplikace spustí simulace a výsledné simulace bude serializovat do zadaného souboru

**UC12 – Zobrazení backtestu párové investiční strategie**

1. Aplikace zobrazí okno pro načtení párových investičních strategií
2. Uživatel vybere, ze kterého souboru chce načíst párové investiční strategie
3. Aplikace deserializuje a načte data do tabulky

**UC13 – Spuštění optimalizace párové investiční strategie**

1. Aplikace zobrazí okno pro zadání následujících informací
   a) Název souboru se serializovanými daty
   b) Tickery dat

```
c) Typ aktiva
d) TF aktiva
e) Název strategie včetně rozsahu parametrů
```

2. Uživatel přidá zadané informace do exekučního plánu
3. Uživatel potvrdí spuštění optimalizací podle exekučního plánu
4. Aplikace spustí optimalizace a výsledky bude serializovat do zadaného souboru

**UC14 – Zobrazení optimalizace párové investiční strategie**

1. Aplikace zobrazí okno pro načtení párových investičních strategií
2. Uživatel vybere, ze kterého souboru chce načíst optimalizace
3. Aplikace deserializuje a načte data do tabulky

#### 3.1.4 High-level pohled na architekturu

Celá aplikace je rozdělena do tří vrstev. Nejvyšší vrstva je prezentační, která tvoří uživatelské
rozhraní. Tato vrstva se stará o interakci s uživatelem aplikace. Nejdůležitější vrstvou je
aplikační vrstva, ve které je implementována logika aplikace. V této vrstvě, se nacházejí entity
popisované v teoretické části bakalářské práce. Jsou zde implementovány investiční strategie,
trhy a způsoby aplikace investičních strategií na trhy, čili backtest a optimalizace. Poslední
vrstvou je perzistentní vrstva, která se stará o ukládání a získání dat z úložiště.

```
Obrázek 12: 3-vrstvá architektura
```

```
Zdroj: VS2010, Vlastní zpracování
```

Celé řešení představené v bakalářské práci bylo pojmenováno jako **CSAnalyzer**. Obrázek níže
zobrazuje komunikaci okolí s aplikací. V požadavcích máme definováno, že aplikace musí

umožňovat uživateli stahovat data ze serverů IQFeed. Tato komunikace je naznačena v levé
horní části obrázku, kdy uživatel definuje CSAnalyzeru, jaká data chce stáhnout a kam.
CSAnalyzer, pak předá žádost programu IQFeed client, který data vrátí v požadovaném
formátu. Dalším požadavkem na aplikaci je napojení na statistický software Wolfram
Mathematica 8.0. Tento požadavek je znázorněn na pravé horní straně obrázku. Uživatel
spustí libovolný test, čímž se aplikuje vybraná investiční strategie na trh, s tím, že uvnitř
investiční strategie je zajištěna komunikace s Wolfram Mathematica 8.0. Pokud tedy nastane
potřeba výpočtu, můžeme spoléhat na metody a skripty programu Wolfram Mathematica.

```
Obrázek 13: Popis komunikace s CSAnalyzer
```

```
Zdroj: Vlastní zpracování
```

Z důvodů snadné rozšiřitelnosti byla aplikace rozdělena do několika logických celků. Tyto
logické celky jsou rozděleny podle 3-vrstvé architektury a shromažďují v sobě logicky
související třídy a funkčnosti. Toto rozdělení aplikace odpovídá požadavku na aplikaci, aby
byla snadno rozšiřitelná, stačí tedy přidat do řešení nový logický celek či vyměnit stávající za
jinou implementaci.

```
Obrázek 14: Rozdělení aplikace do vrstev a logických entit
```

```
Zdroj: VS2010, Vlastní zpracování
```

V prezentační vrstvě můžeme vidět entitu CSAnalyzerWinForm, která uživateli zpříztupňuje
přes GUI stanovené funkčnosti programu. Aplikační vrstva patří k nejdůležitější části aplikace.
Jsou zde zastoupeny všechny objekty, které jsme si představili v teoretické části bakalářské
práce. Jsou zde objekty reprezentující trhy stažené ze serverů IQFeed, investiční analýzy,
investiční strategie i způsoby aplikace investičních strategií na trhy. Jelikož jsou tyto objekty
logicky rozdílné a každý objekt řeší pouze část problému, jsou i tyto objekty rozděleny do

různých logických celků. V perzistentní vrstvě se nachází pouze jeden logický celek
CSSerializer, který řeší ukládání a načítání objektů z aplikační vrstvy. Ukládání a načítání dat je
v této entitě realizováno serializací/deserializací, což nám zaručuje jednoduchost použití,
rychlost načítání/ukládání dat a možnost přenášet objekty mezi počítači. V následující kapitole
si představíme konkrétní implementaci navržených entit a komunikací.

### 3.2 Implementace

V této kapitole představíme použité technologie v aplikaci a popíšeme si implementaci
jednotlivých logických celků představených v kapitole 3.1 Analýza a návrh. Každá
implementace jednotlivého celku je strukturována do samostatné kapitoly.

#### 3.2.1 Zvolené technologie

Aplikace byla implementována za pomocí **.NET Frameworku 4.0** , objektově orientovaného
programovacího jazyka **C#** a GUI aplikace za pomocí WinForms, jelikož se nekladl důraz na
použitý způsob prezentace. Díky použití **WinForms** jsme docílili jednoduchého a efektivního
GUI, který lze v rámci aplikace rozšířít či dokonce vyměnit za některý z ostatních způsobů
realizace GUI. V rámci implementace byl použit způsob otevírání oken pomocí **daemon
vláken** , které nám zaručují možnost používání aplikace i při vysokém zatížení jednoho z oken.
Vzhledem k proměnlivému počtu parametrů pro různé investiční strategie byl použit princip
**reflexe** , abychom splnili požadavek na snadné rozšíření aplikace. Kdybychom tedy nezvolili
princip reflexe, musel by v aplikaci existovat pro každou investiční strategii zváštní formulář
na vyplnění parametrů strategie. V prezentační vrstvě se pro zobrazování grafů využívá objekt
**Chart** z jmenného prostoru System.Windows.Forms.DataVisualization. Díky tomuto objektu
jsme v aplikaci schopni vytvářet jednoduché grafy, které jsou důležité pro grafické znázornění
některých statistických veličin a na zobrazení OHLC dat pro stažená aktiva. Některé grafy ale
tímto objektem nevytvoříme, museli jsme proto využít grafů z další aplikace. Touto důležitou
technologií použitou při implementace aplikace je **Wolfram.NETLink**^11 , který nám umožní
komunikovat s programem Wolfram Mathematica 8.0. Tuto komunikaci budeme využívat
hlavně pro statistické výpočty a tvorbu některých grafů. Stahování dat z IQFeed pak probíhá
za pomocí síťového protokolu **TCP/IP a socketu**. V prostředí .NET je práce se síťovými
protokoly a se sockety realizována pomocí jmenného prostoru System.Net. Pro získávání a
ukládání dat je pak využit princip **serializace** a **deserializace** v prostředí .NET.
11
Více 3.2.3 Entity aplikační vrstvy

#### 3.2.2 Entity prezentační vrstvy

V prezentační vrstvě jsou všechny formuláře seskupeny pod entitu s názvem
CSAnalyzerWinForm. Důvodem existence jedné jediné entity v prezentační vrstvě je fakt, že
jedním z požadavků na aplikaci je její jednoduché rozšíření. V tomto směru lze do projektu
pouze přidat novou entitu, a to například CSAnalyzerWPF, která by nám reprezentovala GUI
navržené ve WPF technologii.

```
Obrázek 15: Prezentační vrstva a interakce s okolím
```

```
Zdroj: VS2010, Vlastní zpracování
```

Obrázek výše naznačuje interakci uživatele aplikace s entitou CSAnalyzerWinForm, kdy
uživatel využívá GUI této komponenty pro práci s ostatními entitami ostatních vrstev. V
rámci entity jsou však formuláře rozděleny do logických celků podle toho, které funkčnosti
dané formuláře řeší. Stejně jako jsou rozděleny entity do logických celků v rámci aplikační
vrstvy^12 , jsou rozděleny i formuláře v prezentační vrstvě. V aplikační vrstvě se však jednalo o
rozdělení do entit, kdežto v prezentační vrstvě se jedná o rozdělení do package (neboli složek).
Prvním logickým celkem je **DataManager** , který nám obsluhuje entitu CSDataManager v
aplikační vrstvě. Ta nám umožňuje stahovat data z IQFeed. Stahování dat v této entitě si
představíme v následující kapitole.

12
viz Obrázek 14: Rozdělení aplikace do vrstev a logických entit

```
Obrázek 16: DataManager package
```

```
Zdroj: VS2010, Vlastní zpracování
```

Dalším logickým celkem je **DataAnalyzer** , který nám umožňuje spouštět historické simulace a
optimalizace. Přesněji nám umožňuje obsluhovat objekty v entitě CSAnalyzerCore a
CSDataAnalyzer v aplikační vrstvě. Ve formulářích celku DataAnalyzer tedy vyplňujeme
odkud se mají vzít data aktiv, jaký timeframe tyto data mají a jakou strategii budeme chtít
aplikovat. Před samotnou aplikací je nutné zadat nejdříve vstupní parametry pro tuto strategii.
A pokud se bude jednat o optimalizaci, musíme zadat také rozsah těchto parametrů včetně
krokovací proměnné.

```
Obrázek 17: DataAnalyzer package
```

```
Zdroj: VS2010, Vlastní zpracování
```

Pro zobrazování dat, simulací a optimalizací slouží package **DataViewer**. Pokud tedy budeme
chtít nahlížet na data v úložišti, budeme využívat formuláře z této package. Tyto formuláře
jsou pak rozděleny podle stejných základů jako v teoretické části. Tedy na tržní a párové
formuláře zajišťující historickou simulaci nebo optimalizaci.

```
DataAnalyzer
CSMA_BacktestForm
Attributes
Operations
CSMS_BacktestForm
Attributes
Operations
CSMS_OptimalizationForm
Attributes
Operations
```

```
CSPA_BacktestForm
Attributes
Operations
```

```
CSPS_BacktestForm
Attributes
Operations
```

```
CSPS_OptimalizationForm
Attributes
Operations
```

```
Obrázek 18: DataViewer package
```

```
Zdroj: VS2010, Vlastní zpracování
```

V úložišti však může být uloženo i velké množství dat, simulací či optimalizací. Je proto
vhodné rozdělit zobrazování všech informací do více formulářů. Nebylo by tedy vhodné při

```
DataViewer
```

```
CSMA_BacktestsForm
Attributes
Operations
```

```
CSMS_BacktestsForm
Attributes
Operations
```

```
CSMA_BacktestDetailForm
Attributes
Operations
```

```
CSMS_BacktestDetailForm
Attributes
Operations
```

```
CSPA_BacktestsForm
Attributes
Operations
CSPA_BacktestDetailForm
Attributes
Operations
```

```
CSPS_BacktestsForm
Attributes
Operations
CSPS_BacktestDetailForm
Attributes
Operations
```

```
CSMS_OptimalizationsForm
Attributes
Operations
CSMS_OptimalizationDetailForm
Attributes
Operations
```

```
CSPS_OptimalizationsForm
Attributes
Operations
CSPS_OptimalizationDetailForm
Attributes
Operations
```

```
MarketsForm
Attributes
Operations
MarketsDetailForm
Attributes
Operations
```

načítání všech simulací do formuláře zobrazovat veškeré informace, které máme k dispozici.
Tento formulář zobrazuje obrázek níže.

```
Obrázek 19: Formulář backtesty tržních strategií
```

```
Zdroj: [14], Vlastní zpracování
```

Navíc, pokud budeme načítat z úložiště velké množství dat, je vhodné tato data filtrovat,
abychom omezili načítání jen na takova data, která chceme opravdu vidět. Není totiž žádoucí z
hlediska výkonu, abychom načítali všechna data. V těchto filtrech je možné spatřit i možnost
zobrazení výsledků simulace pouze za určité období (neboli časový úsek) a v určitém typu
výnosu. Tímto navazujeme na teoretickou část z kapitoly 2.4 Analýza výsledků.

Tento **obecný formulář** slouží především pro zobrazení všech provedených simulací, které
chceme zobrazit (neboli budou vyhovovat filtru). Pokud ale budeme chtít vidět více informací
o provedené simulaci, slouží nám k tomu **detailní formulář** , který otevřeme dvojkliknutím na

řádek simulace. Tento detail nám pak zobrazí i ostatní analýzy a veličiny, které můžeme
vypočítat ze simulace na základě kapitoly 2.4 Analýza výsledků.

```
Obrázek 20: Detailní formulář backtestu tržní strategie
```

```
Zdroj: [14], Vlastní zpracování
```

Tímto způsobem jsou implementovány všechny formuláře pro historickou simulaci tržní a
párové analýzy. Vyjímkou nejsou ani historické simulace pro tržní a párové strategie. Odlišně
pak vypadají pouze formuláře pro optimalizaci tržních a párových strategií. V těchto
formulářích lze nalézt pouze výpis provedených backtestů v počtu, který byl zadán při
inicializaci optimalizace. Jak již víme z teoretické části, v rámci optimalizace dochází k výběru
stanoveného počtu backtestů na základě jedné vybrané statistiky.

#### 3.2.3 Entity aplikační vrstvy

Aplikační vrstvu jsme rozčlenili do několika entit, z nichž každá řeší elementární úlohy podle
požadavků na aplikaci. Z teoretické části jsme již identifikovali nejdůležitější objekty a v této
vrstvě tyto entity začleníme podle logického rozdělení do entit. Jedním z požadavků bylo

umožnění stahování data z IQFeed a komunikace s Wolfram Mathematica,
naznačeny na obrázku níže.

```
Obrázek 21 : Aplikační vrstva a interakce s okolím
```

umožnění stahování data z IQFeed a komunikace s Wolfram Mathematica,
umožnění stahování data z IQFeed a komunikace s Wolfram Mathematica, jenž jsou

```
Zdroj: VS2010, Vlastní zpracování
```

**CSAnalyzerCore** obsahuje **implementaci** objektů reprezentující **aktiva** a jejich data. Tato
aktiva mají společného předka AktivumBase, který slouží k tomu, abychom se nemuseli starat
o to, s jakou konkrétní implementací pracujeme. Usnadní nám práci s objekty aktiv a
pohodlněji se nám pak bude manipulovat i s daty. Další výhodou je snadná rozšiřitelnost
aplikace o nová aktiva. Jelikož se nám nenaruší práce s aktivy v ostatních formulářích ani v
rámci používání v ostatních logických celcích.

```
Obrázek 22: Implementace trhů v CSAnalyzer
```

```
Zdroj: VS2010, Vlastní zpracování
```

Mimo aktiv obsahuje CSAnalyzer také **implementaci investičních analýz** a strategií.
Investičními analýzami rozumíme indikátory^13 , ze kterých se skládají pravidla pro investiční
strategii. V praxi se však můžeme setkat s tak sofistikovanou investiční analýzou, která v sobě
může kombinovat několik dalších investičních analýz. Je proto důležité tyto objekty navrhnout
tak, aby se daly využívat v investičních strategiích, ale zároveň abychom mohli tvořit
komplexní investiční analýzy. Dalším důležitým faktem je použitelnost investičních analýz,
proto jsme museli definovat společného předka pro všechny investiční analýzy, který by nám
stejně jako u trhů zaručil, že budeme moci pracovat jednotně se všemi konkrétními
implementacemi investičních analýz. Samozřejmě, stejně jako u trhů, tímto splňujeme
požadavek na snadné rozšíření, jelikož po přidání nové analýzy, nebudeme muset editovat
používané formuláře, ani způsoby aplikace na trhy. K implementaci tohoto žádoucího chování
použijeme **design pattern Strategy**. Tento pattern zapouzdřuje skupinu algoritmů, aby byly
vzájemně zaměnitelné. Vzájemná zaměnitelnost nastává v případě, že všechny algoritmy
přepíší zděděnou metodu z předka. Díky tomu mohou klienti pracovat s algoritmy nezávisle
na konkrétní implementaci této metody. Metodou, která je společná pro všechny

13
Viz 2.2 Investiční strategie

implementace, je metoda newBar(), která je na předcích obou druhů analýz a obou druhů
strategií.

```
Obrázek 23: Strategy pattern investičních analýz v CSAnalyzer
```

```
Zdroj: VS2010, Vlastní zpracování
```

Z teoretické části bakalářské práce již známe pravidla všech tří vybraných investičních
strategií^14 a známe také jejich rozdělení do skupin^15. Toto rozdělení musíme brát v potaz i při
implementaci analýz, proto jsme rozdělili všechny potřebné analýzy také na **tržní a párové**^16.
Implementace strategy patternu pro párové analýzy je stejná jako u tržních analýz na obrázku
18., rozdíl je jen v pojmenování předka. Při implementaci párových analýz jsme také museli
brát v potaz to, že musíme používat a pracovat se dvěmi trhy najednou, což je rozdíl oproti
tržním analýzám, kde pracujeme pouze s daty jednoho trhu. Lišit se tedy jednotlivé metody
budou zejména ve vstupních parametrech.

Implementace strategy patternu u investičních strategií se v podstatě neliší od implementace u
investičních analýz. Rozdělení strategií je také totožné s rozdělením u investičních analýz.
Rozlišujeme tedy strategie tržní a párové. Rozdíl mezi analýzou a strategií tkví v tom, co
očekáváme jako výsledek. V případě aplikace analýzy na tržní data je výsledkem pole hodnot
této analýzy. Vstupem jsou tedy data aktiva, pak následuje aplikace pravidel analýzy a
výstupem je pole hodnot. V případě investiční analýzy klouzavého průměru je výsledkem pole
hodnot průměrů posouvaných v čase^17. Naopak u investiční strategie jsou vstupem tržní data,

14
15 Viz 2.2 Investiční strategie
16 Viz 2.2.1 Tržní strategie, 2.2.2 Párové strategie
17 Podle 2.2.1 a 2.2.2
Podle Obrázek 1 : Vzorec klouzavého průměru

pak následuje aplikace pravidel strategie a výstupem je pole hodnot signálů [7, s.15]. V rámci
aplikace pravidel strategie musí samozřejmě docházet k přepočtu analýz obsažených v této
strategii.
Obrázek 24: Vstup a výstup investiční strategie v CSAnalyzer

```
Zdroj: [7, s.17, obr.1.1]
```

Polem signálů je myšleno pole hodnot 1,0 a -1. Hodnota 1 nám značí stav, kdy je strategie v
dlouhé pozici. Respektive nám značí stav, kdy jsou splněna pravidla strategie pro dlouhou
pozici. Hodnota 0 pak značí stav, kdy strategie není v žádné pozici. Respektive nám značí stav,
kdy nejsou splněna pravidla ani pro dlouhou ani pro krátkou pozici. Hodnota -1 pak značí
krátkou pozici. Stejně jako u implementace investičních analýz musíme při aplikaci investiční
strategie počítat s tím, že u párové varianty pracujeme s dvěmi trhy najednou a generujeme
signály na dva trhy současně, kdežto u tržní varianty pracujeme pouze s jedním trhem a
generujeme signály pro jeden trh.

Další entitou v aplikační vrstvě je **CSDataAnalyzer**. Tato entita implementuje objekty pro
aplikaci investičních strategií a analýz na trhy^18. Samotné investiční analýzy a strategie
neobsahují aparát pro historickou simulaci, obsahují pouze pravidla, jak se má v učitých
okamžicích reagovat nebo jaký matematický vzorec se má na data aplikovat. Jak jsme již
uvedli, hlavní metoda pro analýzy a strategie je newBar(). Tato metoda má však jako vstupní

18
Viz 2.3 Aplikace investičních strategií

parametry pouze aktuální datum, aktuální otevírací cenu (neboli open), aktuální nejvyšší cenu
(neboli high), aktuální nejnižší cenu (neboli low) a aktuální zavírací cenu (neboli close). V
případě párové analýzy nebo strategie jsou vstupními parametry aktuální datum, aktuální
otevírací cena na aktivu 1 (neboli open 1), aktuální nejvyšší cenu na aktivu 1 (neboli high 1),
aktuální nejnižší cenu na aktivu 1 (neboli low 1), aktuální zavírací cenu na aktivu 1 (neboli
close 1), aktuální otevírací cena na aktivu 2 (neboli open 2), aktuální nejvyšší cenu na aktivu 2
(neboli high 2), aktuální nejnižší cenu na aktivu 2 (neboli low 2) a aktuální zavírací cenu na
aktivu 2 (neboli close 2). Pokud tedy analýza vyžaduje k výpočtu i starší ceny, musí si je sama
evidovat uvnitř třídy. Tento postup je velice žádoucí, protože nechceme, aby si analýza sahala
pro starší data do úložiště pokaždé, když bude pořebovat přepočítat svoji výslednou hodnotu.
Tato situace nastane pokaždé, když analýze při historické simulaci předáme nové údaje.

Rozdělení v entitě CSDataAnalyzer probíhá podle stejných teoretických základů jako u
CSAnalyzerCore. Rozlišujeme tedy tržní a párové aplikace investičních analýz a strategií. Je to
hlavně proto, že pokud jsou rozdílné samotné analýzy a strategie, které vyžadují rozdílné
vstupní argumenty v metodě newBar(), musejí existovat take rozdílné aplikace pro tržní,
párové analýzy a strategie. Další rozdělení je již známé z kapitoly 2.3 Aplikace investičních
strategií v teoretické části bakalářské práce. Aplikaci investičních strategií na aktiva
rozdělujeme na backtest (neboli historickou simulaci) a optimalizaci. Způsob implementace
historické simulace je zobrazen na obrázku níže.

```
Obrázek 25: Implementace backtestu v CSAnalyzer
```

```
Zdroj: VS2010, Vlastní zpracování
```

**Backtest** je historická simulace, díky které jsme schopni zjisit výkonnost analýzy či strategie
na historických datech. V teoretické části práce jsme popsali, co to je historická simulace
detailněji, popsali jsme k čemu se historická simulace využívá a co je pro ní potřeba. Na
následujícím obrázku si představíme kroky, které se při historické simulaci provádějí v aplikaci
CSAnalyzer. Oproti teoretické části jsou zde navíc některé kroky, které nám usnadňují
následnou práci s backtesty. Není tedy cílem pouze provést historickou simulaci, ale také ji
připravit na následné použití v programu.

```
Obrázek 26: Sekvenční diagram backtestu v CSAnalyzer
```

```
Zdroj: VS2010, Vlastní zpracování
```

Protože investiční analýzy a strategie přepisují metodu newBar() z předka, lze toho využít při
aplikaci. V konstruktoru těchto historických simulací předáme objektu analýzy nebo strategie,
kterým chceme provést historickou simulaci. Pak zavoláme metodu runBars_Backtest(), které
předáme jako vstupní parametry pole hodnot dat, otevíracích cen, nejvyšších cen, nejnižších
cen a uzavíracích cen. V případě párové historické simulace pak předáváme pole hodnot dvou
aktiv namísto jednoho. Objekt historické simulace pak v metodě runBars_Backtest() projde
všechna data a postupně předá objektu analýzy či strategie ceny svázané s tímto datem.
Jakmile je aplikace/simulace hotova, tj. když analýza nebo strategie dostala všechny dostupné
bary, je spuštěn výpočet reportů. Tyto reporty nám pomáhají při filtraci výsledků ve

```
this : CSMS_BacktestForm this.DB_Serializer : SerializerBase CSMS_SMAcrossSystemstrategyTemp : Backtester : MarketBacktester
button6_Cl...
getETF
<<return>>
getFuture
<<return>>
Create CSMS_SMAcrossSystem
```

```
<<return>>
```

```
Create MarketStrategyBase
Create CSMA_SMA
Create CSMA_SMA
Create MarketBacktester
<<return>>
```

```
CloneClean
runBars_Backtest
```

```
<<return>>
```

```
newBar
```

```
finalize
fillQuickReports
```

```
saveMarketBacktester
<<return>>
Dispose
<<return>>
```

```
Dispose
```

```
[for (int i = 0;i <= aktivum_DateTimes.Length - 1; Loop i++)]
```

```
If
[if (strategy != null)]
```

formulářích. Kdybychom tedy měli v úložišti uloženo přes tisíc backtestů, lze si zobrazit jen
takové, které svými výsledky odpovídají našim představám. Důvodem, proč však jsou tyto
reporty generovány ihned po historické simulaci, je ten, že pokud bychom generovali reporty
až při dotazování z formulářů, ztráceli bychom drahocenný čas, jelikož bychom museli čekat
nejenom než se nám všechny strategie a analýzy profiltrují, ale také než se vygenerují všechny
potřebné reporty. Dalším důvodem, proč generovat reporty hned po dokončení simulace je
existence požadavku, díky kterému aplikace umožní uživateli ukládat pouze backtesty
vyhovující zadaným podmínkám^19. Proto je zapotřebí mít k dispozici reporty ihned po
dokončení backtestu, abychom mohli uložit do úložiště pouze ty backtesty, které vyhovují
našim představám. Výhodné je to také proto, že nemusíme v úložišti zbytečně evidovat
strategie s nežádoucími výsledky.

**Optimalizace** je způsob, jakým získat představu o výkonnosti strategie napříč zadanými
parametry strategie. Jedná se o vhodný způsob jak zjistit ty nejvhodnější parametry a jak získat
představu o robusnosti strategie z pohledu zadávaných parametrů.

```
Obrázek 27: Implementace optimalizace v CSAnalyzer
```

```
Zdroj: VS2010, Vlastní zpracování
```

Způsobů implementace optimalizací je mnoho, od jednoduché optimalizace, která provede
historickou simulaci pro každý zadaný parameter, až po genetickou optimalizaci, která provede
historickou simulaci pouze ve vybraných větvích a následně pomocí genetických algoritmů
testuje již jen v rámci vybrané větvě. My jsme zvolili princip jednoduché optimalizace^20 (neboli
Exhaustive optimalizace), jelikož se jedná o způsob, který nám otestuje každou kombinaci
parametrů a my tak můžeme získat úplnou představu o výkonnosti strategie. Jedná se o
vhodný způsob pro implementaci, jelikož nám nabízí jednoduché, avšak účinné zpracování
informací. V určitých případech ale můžeme chtít optimalizovat velice komplexní strategii,
která bude obsahovat velké množství analýz. Tím pádem bude mít i strategie velké množství
vstupních parametrů a optimalizace této strategie by byla časově náročná. V tomto případě
pak lze za vhodnější způsob optimalizace považovat právě genetickou optimalizaci.

19
20 viz 3.1.1 Funkční požadavky
viz 2.3.2 Optimalizace

Obrázek 28: Sekvenční diagram optimalizace v CSAnalyzer

```
Zdroj: VS2010, Vlastní zpracování
```

Při optimalizaci využíváme objektů investičních strategií, které postupně iniciujeme pro každý
zadaný parametr. Každou iniciovanou strategii vkládáme do pole (listu) a celé toto pole pak
předáme objektu optimalizace. Objekt optimalizace postupně otestuje (provede historickou
simulaci) všechny kombinace zadaných parametrů a do paměti si ukládá jen report z této
historické simulace. Jakmile máme uloženy v paměti všechny reporty, jsme schopni tyto
reporty seřadit podle námi zadaného výkonostního parametru a vybrat pouze stanovený počet
vyhovujících reportů. Z těchto reportů se pak zpětně dozvíme, k jakým parametrům se tyto
reporty vztahují. A díky tomu spustíme znovu historickou simulaci, ale už jen pro ty
parametry, které nám v reportech vyšly jako vyhovující. Nakonec provedené historické
simulace uložíme v objektu optimalizace do pole vyhovujících simulací a celý objekt
optimalizace uložíme do úložiště.

Entitou zajišťující komunikaci s IQFeed je **CSDataManager**. IQFeed je zpoplatněná služba
poskytující historická data ve formátu OHLC. Po zaregistrování máme k dispozici možnost
stáhnout si program IQFeed client, který je pro naši komunikaci s IQFeed klíčový.
Dokumentace, ke které máme po zaregistrování také přístup, nám dává na výběr, jak
komunikovat s programem IQFeed client. Způsoby komunikace jsou dva. První možností je
komunikace přes COM objekty. Druhou a námi zvolenou je pak komunikace přes TCP/IP
protokol. My jsme tedy pro naši komunikaci zvolili síťový protokol TCP/IP a jeho
reprezentaci v .NET pomocí Socket třídy v jmenném prostoru System.Net. Tento způsob
nám umožňuje efektivně se dotazovat na data a následně je po částech dostávat jako odpověď
[9].

Obrázek níže popisuje kroky při stahování dat v programu CSAnalyzer. Prvním krokem je
vytvoření a pojmenování souboru s daty. V dalším kroku vytvoříme objekt DataDownloader,
který implementuje logiku stahování dat. Tomuto objektu předáme seznam aktiv, která
chceme stáhnout, včetně důležitých informacích o aktivu. Tyto informace jsou sdruženy a
udržovány v číselníku aplikace CSAnalyzer. DataDownloader se poté pokusí spojit přes
TCP/IP protokol s aplikací IQFeed. Pokud se spojení podaří, vytvoří první aktivum, které má
v seznamu. Během vytvoření aktiva dojde k naplnění důležitými informacemi (vlastnostmi) z
číselníku. Následně odešle žádost o všechny OHLC data pro dané aktivum v zadaném
rozmezí dat. IQFeed client postupně vrátí OHLC data pro zadané aktivum a DataDownloader
postupně naplní toto aktivum vrácenými daty. Jakmile vrácená data obsahují konečný tag,
DataDownloader aktivum uzavře a uloží. Tento proces se opakuje pro každé zadané aktivum,
dokud nejsou všechna aktiva ze seznamu stažena a uložena.

Obrázek 29: Sekvenční diagram stahování dat v CSAnalyzer

```
Zdroj: VS2010, Vlastní zpracování
```

Dalším požadavkem na aplikaci je, že musíme umožnit komunikaci se statistikým softwarem
Wolfram Mathematica 8.0. Ta je zapouzdřena v entitě **CSWolframMathematica**.
Komunikace probíhá pomocí Wofram.NETLink [20], který nám umožňuje nejenom volat
metody Wolfram Mathematica 8.0, ale také nám umožňuje volat námi vytvořené scripty,
vytvořené pomocí Wolfram Workbench 2.0 [8].

```
Obrázek 30: Sekvenční diagram volání existující metody v Wolfram Mathematica 8.0
```

```
Zdroj: VS2010, Vlastní zpracování
```

Jedním ze způsobů jak volat metody aplikace, je přes objekty ILoopbackLink a Expr, které
nám poskytuje knihovna Wolfram.NETLink. Jakmile vytvoříme objekt ILoopbackLink,
můžeme mu předat vstupní parametry, které potřebujeme pro výpočty. Následně z tohoto

objektu pomocí metody GetExpr() získáme objekt Expr, který nám reprezentuje náš příkaz.
Ten pak vložíme do metody Evaluate() objektu MathKernel. Objekt MathKernel je pro nás
velice důležitý, jelikož nám umožňuje komunikovat s Wolfram Mathematica 8.0 pomocí
Kernel aplikací reprezentujících jádro aplikace běžící v samostatném vlákně. Obrázek 25
popisuje volání již existujících metod v programu Wolfram Mathematica 8.0.

Na druhou stranu, v některých případech nechceme pouze volat již existující metody.
Potřebujeme napsat vlastní metodu, která bude vytvářet analýzy, či řešit určitou část
problémové domény. Pro tyto situace využijeme software Wolfram Workbench 2.0, ve kterém
vytvoříme vlastní script pomocí scriptovacího jazyka Wolfram Mathematica 8.0 [8, 19]. V
našem řešení jsme museli implementovat dvě vlastní metody, které nám pokrývají potřeby
aplikace CSAnalyzer. V prvním případě se jedná o metodu GetExpectedFull(), která nám vrací
numerické výpočty a obrázky pro detailní formuláře. V těchto formulářích máme část, kterou
považujeme za budoucí hodnoty, ty se vypočítávají právě díky námi implementované metodě
ve Wolfram Workbench 2.0. Druhým případem je GetLinearRegressionAnalyse(), která nám
analyzuje vývoj equity křivky z pohledu lineární regrese podle teoretické části bakalářské
práce^21.

#### 3.2.4 Entity datové vrstvy

V perzistentní vrstě je pouze jedna entita **CSSerializer** , která nám zajištuje ukládání a načítání
dat ze souborů. Interakce databázové vrstvy s okolím je naznačena na obrázku níže.

```
Obrázek 31: Perzistentní vrstva a interakce s okolím
```

Zdroj: VS2010, Vlastní zpracování
21
Viz 2.4.1 Popisná statistika

Implementace úložiště pro aplikaci CSAnalyzer byla realizována za pomocí principu serializace
a deserializace. Samotné objekty využívají abstraktní třídu k tomu, aby schovali vlastní
implementaci. Tento princip splňuje požadavek kladený na aplikaci, a to aby umožnila
jednochou rozšišiřitelnost^22. Příkladem rozšíření může být v tomto případě nová třída
zajišťující serializaci a deserializaci do vzdáleného úložiště, nyní je implementována lokální
**serializace** a **deserializace**.

```
Obrázek 32: Implementace datové vrstvy v CSAnalyzer
```

```
Zdroj: VS2010, Vlastní zpracování
```

Serializace je způsob, jak uchovat instance objektů po delší dobu v perzistentním úložišti.
Principem serializace je převedení instance objektu na **stream** (neboli proud), díky čemuž
může být uložen do souboru. Výhodou převedení na stream je i fakt, že stream může být
poslán přes HTTP protokol nebo nutnost existence serializace pri využívání .NET remotingu,
kdy používáme objekty mezi různými aplikačními doménami [22]. Vybraný způsob serializace
byl zvolen pomocí **BinaryFormatter** , který umožňuje ukládat instance v binární podobě [21].

Stream stream = File.Open(DBpath + TempDBName, FileMode.Append);
BinaryFormatter bFormatter = new BinaryFormatter();
bFormatter.Serialize(stream, backtester);
stream.Close();

Jelikož se jedná o uložení binární podoby objektu do souboru, použili jsme **FileStream** pro
konverzi objektu na stream [21]. Ten samý princip je použit i u deserializace, jelikož není
možné serializovat objekty do souboru pomocí BinaryFormatteru a deserializovat je jiným
způsobem.

22
viz 3.1.2 Nefunkční požadavky

using (var fileStream = new FileStream(DBpath + TempDBName, FileMode.Open))
{
var bFormatter2 = new BinaryFormatter();
while (fileStream.Position != fileStream.Length)
{
MarketBacktester tempBacktester = (MarketBacktester)
bFormatter2.Deserialize(fileStream);
}
}

Všechny objekty, které chceme serializovat a deserializovat pomocí komponenty CSSerializer,
musí být označeny atributem **Serializable**.

[Serializable]
public class MarketBacktester : IDisposable
{
}

V následující kapitole si představíme, jak využít implementované aplikace CSAnalyzer při
realizaci out-of-sample testu.

### 3.3 Aplikace

V této kapitole představíme způsob, jak využít aplikace CSAnalyzer pro out-of-sample test.
Problémem, který nastává při testování na různých trzích a různých strategií, je způsob, jak
zaručit porovnatelnost dosažených výsledků. Pokud totiž testujeme více strategií na více
trzích, nastává problém s investovanou částkou. Každý trh může mít různou velikost ticku a
také dolarovou hodnotu tohoto ticku. Pokud tedy strategie budou nakupovat a prodávat vždy
100 kusů (neboli bude kusově neutrální) a hodnota ticku (dolarová či jednotková) se bude pro
dané trhy lišit. Nemůžeme zajistit správnou porovnatelnost mezi aktivy. Na různých trzích s
různou cenou a různou hodnotou ticku, budeme při nákupu 100 kusů dosahovat různých
výnosů v USD. Jedná se tedy o **problém investované částky** a **problém cen aktiv** (osa _Y_ ).
Tyto problémy vyřešíme tak, že stanovíme investovanou částku všech strategií na 1000 $
(neboli budeme dolarově neutrální) a dosažené výsledky strategií budeme porovnávat v
relativní míře. Jako relativní míru jsme zvolili logaritmický výnos, jelikož je pro použítí
vhodnější než aritmetický výnos^23. Další **problém** je **různá délka aktiv**. Ta způsobí, že
budeme dosahovat různých výnosů na aktivech s různou délkou dat. Pokud tedy chceme
23
Více 2.4 Analýza výsledků

sjednotit výnosy z hlediska času (osa _X_ ), musíme použít annualizovaný výnos^24. Ten nám
zaručí porovnatelnost výnosů strategií na různých trzích.

#### 3.3.1 Trhy

Vybrané trhy jsme stáhli ze serverů IQFeed a timeframe (dále jen TF) byl zvolen hodinový
(1H). Rozdělili jsme je pro out-of-sample testování v poměru **zhruba 75% a 25%** , což
odpovídá rozdělení pro IS optimalizaci od 4.5.2007 do 1.1.2011 a pro OOS backtest od
1.1.2011 do 1.4.2012. Přesné rozdělení pro jednotlivé trhy je zobrazeno v tabulce níže.

```
Tabulka 2: Vybrané trhy ETF a Futures
```

**Trh OptimalizaceIn-Sample**^ **BacktestOut-of-Sample**^ **Dohromady**^

**SPY**

```
13881 barů
(73%)
4.5.2007 - 31.12.2010
```

```
5015 barů
(27%)
1.1.2011 – 1.4.2012
```

```
18897 barů
(100%)
4.5.2007 – 1.4.2012
```

**DIA**

```
11854 barů
(73%)
4.5.2007 - 31.12.2010
```

```
4333 barů
(27%)
1.1.2011 – 1.4.2012
```

```
16188 barů
(100%)
4.5.2007 – 1.4.2012
```

**IWM**

```
12108 barů
(72%)
4.5.2007 - 31.12.2010
```

```
4734 barů
(28%)
1.1.2011 – 1.4.2012
```

```
16843 barů
(100%)
4.5.2007 – 1.4.2012
```

**EWG**

```
7460 barů
(70%)
4.5.2007 - 31.12.2010
```

```
3244 barů
(30%)
1.1.2011 – 1.4.2012
```

```
10705 barů
(100%)
4.5.2007 – 1.4.2012
```

**ZG**

```
11509 barů
(83%)
4.5.2007 - 31.12.2010
```

```
2398 barů
(17%)
1.1.2011 – 1.4.2012
```

```
13908 barů
(100%)
4.5.2007 – 1.4.2012
```

**ZI**

```
9945 barů
(86%)
4.5.2007 - 31.12.2010
```

```
1634 barů
(14%)
1.1.2011 – 1.4.2012
```

```
11580 barů
(100%)
4.5.2007 – 1.4.2012
Zdroj: [14], Vlastní zpracování
```

Jednotlivý řádek popisuje konkrétní trh. V prvním sloupci zjistíme, kolik barů máme k
dispozici pro IS optimalizaci a kolik procent to je v celkovém rozpětí. Další sloupec pak
popisuje ty samé informace pro OOS backtest. Poslední sloupec je shrnutí pro celé období,
kdybychom jej nerozdělili na IS a OOS. U párů sledujeme také **korelaci** , neboli závislost obou
24
Viz 2.4 Analýza výsledků

aktiv pomocí korelačního koeficientu. Ta musí být **alespoň 75%** , abychom mohli považovat
závislost za silnou. Páry jsou vypsány v tabulce níže.

```
Tabulka 3: Vybrané párové trhy ETF a Futures
```

**Pár OptimalizaceIn-Sample**^ **BacktestOut-of-Sample**^ **Dohromady**^

**DIA_EWG**

```
7396 barů
(71%)
4.5.2007 - 31.12.2010
Korelace 95%
```

```
3071 barů
(29%)
1.1.2011 – 1.4.2012
```

```
10467 barů
(100%)
4.5.2007 – 1.4.2012
```

**DIA_IWM**

```
11369 barů
(72%)
4.5.2007 - 31.12.2010
Korelace 95%
```

```
4246 barů
(28%)
1.1.2011 – 1.4.2012
```

```
15615 barů
(100%)
4.5.2007 – 1.4.2012
```

**DIA_SPY**

```
11819 barů
(73%)
4.5.2007 - 31.12.2010
Korelace 99%
```

```
4310 barů
(27%)
1.1.2011 – 1.4.2012
```

```
16129 barů
(100%)
4.5.2007 – 1.4.2012
```

**EWG_DIA**

```
7397 barů
(71%)
4.5.2007 - 31.12.2010
Korelace 95%
```

```
3071 barů
(29%)
1.1.2011 – 1.4.2012
```

```
10468 barů
(100%)
4.5.2007 – 1.4.2012
```

**EWG_IWM**

```
7413 barů
(70%)
4.5.2007 - 31.12.2010
Korelace 85%
```

```
3192 barů
(30%)
1.1.2011 – 1.4.2012
```

```
10605 barů
(100%)
4.5.2007 – 1.4.2012
```

**EWG_SPY**

```
7417 barů
(70%)
4.5.2007 - 31.12.2010
Korelace 96%
```

```
3218 barů
(30%)
1.1.2011 – 1.4.2012
```

```
10635 barů
(100%)
4.5.2007 – 1.4.2012
```

**IWM_DIA**

```
11369 barů
(73%)
4.5.2007 - 31.12.2010
Korelace 95%
```

```
4245 barů
(27%)
1.1.2011 – 1.4.2012
```

```
15614 barů
(100%)
4.5.2007 – 1.4.2012
```

**IWM_EWG**

```
7414 barů
(70%)
4.5.2007 - 31.12.2010
Korelace 85%
```

```
3192 barů
(30%)
1.1.2011 – 1.4.2012
```

```
10606 barů
(100%)
4.5.2007 – 1.4.2012
```

**IWM_SPY** 12055 barů (72%)^ 4716 barů (28%)^ 16771 barů (100%)^

```
4.5.2007 - 31.12.2010
Korelace 95%
```

```
1.1.2011 – 1.4.2012 4.5.2007 – 1.4.2012
```

**SPY_DIA**

```
11819 barů
(73%)
4.5.2007 - 31.12.2010
Korelace 99%
```

```
4310 barů
(27%)
1.1.2011 – 1.4.2012
```

```
16129 barů
(100%)
4.5.2007 – 1.4.2012
```

**SPY_EWG**

```
7415 barů
(70%)
4.5.2007 - 31.12.2010
Korelace 96%
```

```
3218 barů
(30%)
1.1.2011 – 1.4.2012
```

```
10633 barů
(100%)
4.5.2007 – 1.4.2012
```

**SPY_IWM**

```
12056 barů
(72%)
4.5.2007 - 31.12.2010
Korelace 95%
```

```
4716 barů
(28%)
1.1.2011 – 1.4.2012
```

```
16772 barů
(100%)
4.5.2007 – 1.4.2012
```

**ZG_ZI**

```
4984 barů
(90%)
4.5.2007 - 31.12.2010
Korelace 86%
```

```
525 barů
(10%)
1.1.2011 – 1.4.2012
```

```
5509 barů
(100%)
4.5.2007 – 1.4.2012
```

**ZI_ZG**

```
4984 barů
(90%)
4.5.2007 - 31.12.2010
Korelace 86%
```

```
525 barů
(10%)
1.1.2011 – 1.4.2012
```

```
5509 barů
(100%)
4.5.2007 – 1.4.2012
```

```
Zdroj: [14], Vlastní zpracování
```

#### 3.3.2 In-sample optimalizace

Při IS optimalizaci použijeme UC07 a UC13 ze seznamu případů užití aplikace CSAnalyzer
Tyto use-cases (dále jen UC) nám umožní na základě stanovených pravidel výběr jen takových
strategií, které vyhovují námi zadanému filtru. Opět musíme brát v potaz, že řešení celé
aplikace CSAnalyzer je rozděleno na tržní a párové části. Jako první musíme zadat rozsah
parametrů pro **tržní strategii**. Parametry pro strategii SMAcrossover jsou SMAFastPeriod,
SMASlowPeriod, USD4Aktivum a Fee4Aktivum. SMAFastPeriod je parametr pro periodu
SMA indikátoru. SMASlowPeriod je parametr pro periodu dalšího SMA indikátoru. Jedná se
tedy o _n_ ve vzorci 1.1. USD4Aktivum je parametr zastupující investovanou částku a
Fee4Aktivum je parametr nákladu na obchod. Typicky se do tohoto parametru dosazují
poplatky u Brokera. Každý parametr má v optimalizaci tři políčka, první políčko značí, od
které hodnoty se bude parametr optimalizovat, druhé políčko značí hodnotu krokování, neboli
jaká hodnota se bude pokaždé přičítat k parametru při dalším testu. A poslední políčko značí

konečnou hodnotu, pro kterou chceme provést test. Obrázek níže zobrazuje formulář na
zadávání parametrů pro optimalizaci tržní strategie.

```
Obrázek 33: Parametry tržní strategie v CSAnalyzer
```

```
Zdroj: [14], Vlastní zpracování
```

Dalším krokem při optimalizaci je nastavení samotného výběru ze všech provedených testů.
Není totiž žádoucí uchovávat všechny testy z optimalizace, když nás zajímají pouze ty nejlepší
dosažené výsledky z pohledu jednoho či více ukazatelů. V našem případě budeme porovnávat
dosažené výsledky v relativní míře výnosu, tedy podle logaritmického výnosu. Chceme, aby se
nám uchovaly pouze nejlepší dva výsledky, z pohledu RR ratio. Jelikož cílem všech strategií je
mít co největší RR ratio, jedná se o maximalizaci dané veličiny, neboli hledání takových
parametrů strategie, která budem mít co největší RR ratio. RR ratio, jak jsme již uvedly v
teoretické části, je ideální veličina na poměřování výsledků strategií z pohledu výnos vůči
podstupovanému riziku. Tímto způsobem se tedy snažíme nalézt takové parametry strategie,
které dosahují největšího výnosu s nejmenším podstupovaných rizikem. Obrázek níže
zobrazuje formulář pro zadávání parametrů optimalizace v CSAnalyzer.

```
Obrázek 34: Parametry optimalizace v CSAnalyzer
```

```
Zdroj: [14], Vlastní zpracování
```

Po zadání všech tržních strategií na ETF a Futures trhy dostávéme následující exekuční plán.
Exekuční plán je způsob zápisu plánu testů. Jinými slovy tímto způsobem řekneme
CSAnalyzeru, jaké všechny testy/optimalizace chceme provést. Tabulka níže zobrazuje
exekuční plány pro ETF a Futures trhy.

$-etf_is_CSMS ETF | DIA | 1H | CSMS_SMAcrossSystem,10:2:30,50:2:70,1000:1000:1000,2:2:2 | etf_is
ETF | EWG | 1H | CSMS_SMAcrossSystem,10:2:30,50:2:7ETF | IWM | 1H | CSMS_SMAcrossSystem,10:2:30,50:2:70,1000:1000:1000,2:2:2 | etf_is 0,1000:1000:1000,2:2:2 | etf_is
ETF | SPY | 1H | CSMS_SMAcrossSystem,10:2:30,50:2:70,1000:1000:1000,2:2:2 | etf_is
$-futures_is_CSMS Future | @ZG# | 1H | CSMS_SMAcrossSystem,10:2:30,50:2:70,1000:1000:1000,2:2:2 | futures_is
Future | @ZI# | 1H | CSMS_SMAcrossSystem,10:2:30,50:2:70,1000:1000:1000,2:2:2 | futures_is

Parametry pro vybranou párovou strategii jsou MeanRatioPeriod, StdDevPeriod,
USD4Aktivum a Fee4Aktivum. MeanRatioPeriod je parametr pro periodu indikátoru
RatioMean (neboli _SMAPR_^25 ). Jedná se tedy o _n_ ve vzorci 4.1. StdDevPeriod je parametr pro
periodu indikátoru RatioStdDev (neboli _SPR_^26 ). Jedná se tedy o _n_ ve vzorci 5.1. USD4Aktivum
je parametr zastupující investovanou částku na aktivum. Jelikož jsme tedy stanovili
investovanou částku pro strategii na 1000 $, musíme do tohoto parametru zadat 500 $.
Budeme totiž nakupovat dvě aktiva najednou. Obrázek níže zobrazuje formulář na zadávání
parametrů pro optimalizaci párové strategie.

```
Obrázek 35: Parametry tržní strategie v CSAnalyzer
```

```
Zdroj: [14], Vlastní zpracování
```

Parametry optimalizace párové strategie se nijak neliší od parametrů optimalizace tržní
strategie. Chceme porovnávat výnosy v relativní míře, uchovávat dva nejlepší výsledky z
pohledu RR ratio.

25
26 viz vzorec 4^
viz vzorec 5

```
Obrázek 36: Parametry optimalizace v CSAnalyzer
```

```
Zdroj: [14], Vlastní zpracování
```

Po zadání všech tržních strategií na ETF a Futures trhy dostávéme následující exekuční plán.
Exekuční plán je způsob zápisu plánu testů. Jinými slovy tímto způsobem řekneme
CSAnalyzeru, jaké všechny testy/optimalizace chceme provést. V aplikaci CSAnalyzer lze
exekuční plány naklikat pomocí GUI a zobrazených tlačítek. Druhým způsobem je psát
exekuční plány přímo do určeného pole a následně jen spustit testy. Tabulka níže zobrazuje
exekuční plány pro ETF a Futures trhy.

$-etf_is_CSPS
ETF | DIA_EWG | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:2ETF | DIA_IWM | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | etf_is 0,500:500:500,2:2:2 | etf_is
ETF | DIA_SPY | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:2ETF | EWG_DIA | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | etf_is 0,500:500:500,2:2:2 | etf_is
ETF | EWG_IWM | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:2ETF | EWG_SPY | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | etf_is 0,500:500:500,2:2:2 | etf_is
ETF | IWM_DIA | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:2ETF | IWM_EWG | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | etf_is 0,500:500:500,2:2:2 | etf_is
ETF | IWM_SPY | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:2ETF | SPY_DIA | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | etf_is 0,500:500:500,2:2:2 | etf_is
ETF | SPY_EWG | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:2ETF | SPY_IWM | 1H | CSPS_Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | etf_is 0,500:500:500,2:2:2 | etf_is

$-futures*is_CSPS Future | @ZG#*@ZI# | 1H | CSPS*Ratio_2d_0,10:2:20,10:2:20,500:500:500,2:2:2 | futures_is
Future | @ZI#*@ZG# | 1H | CSPS_Ratio_2d_0,10:2:20,1 0:2:20,500:500:500,2:2:2 | futures_is

#### 3.3.3 Out-of-sample backtest

V out-of-sample backtestu využijeme dosažených výsledků z In-sample optimalizace a
vypíšeme nejlepší parametry strategií pro jednotlivé trhy. To jaké parametry vyjdou jako
nejlepší samozřejmě záleží na námi zvoleném kritériu výběru. Pro OOS backtest budeme
využívat UC05 a UC09 ze seznamu případů užití aplikace CSAnalyzer. Následující tabulka

zobrazuje parametry tržní strategie s nejlepšími výsledky z pohledu RR ratio. Tyto parametry
jsme získali z IS optimalizace v aplikaci CSAnalyzer.

```
Tabulka 4: Vybrané parametry tržní strategie z IS optimalizace na ETF a Futures
Trh Parametry
```

```
SPY SMAFastPeriod : 10SMASlowPeriod : 50^
```

```
DIA SMAFastPeriod : 18SMASlowPeriod : 54^
```

```
IWM SMAFastPeriod : 30SMASlowPeriod : 54^
```

```
EWG SMAFastPeriod : 30SMASlowPeriod : 58^
```

```
ZG SMAFastPeriod : 30SMASlowPeriod : 70^
```

```
ZI SMAFastPeriod : 14SMASlowPeriod : 52^
```

```
Zdroj: [14], Vlastní zpracování
```

Podle této tabulky vytvoříme následující exekuční plány pro tržní strategie aplikované na ETF
a Futures trhy v CSAnalyzer. Po vybrání správného databázového souboru s daty, lze tyto
exekuční plány vložit přímo do CSAnalyzeru a spustit testy. Jedná se o stejný princip jako u in-
sample testování. Exekuční plán lze vytvořit přes GUI aplikace nebo jej přímo vepsat do
správného formuláře. UC sloužící ke spustění OS backtestu je v tomto případě UC05 ze
seznamu případů užití.

$-etf_oos_CSMS ETF | SPY | 1H | CSMS_SMAcrossSystem,10,50,1000,2 | etf_oos
ETF | DIA | 1H | CSMS_SMAcrossSystem,18,54,1000,2 |ETF | EWG | 1H | CSMS_SMAcrossSystem,30,58,1000,2 | etf_oos etf_oos
ETF | IWM | 1H | CSMS_SMAcrossSystem,30,54,1000,2 | etf_oos
$-futures_oos_CSMS Future | @ZG# | 1H | CSMS_SMAcrossSystem,30,70,1000,2 | futures_oos
Future | @ZI# | 1H | CSMS_SMAcrossSystem,14,52,1000,2 | futures_oos

Následující tabulka zobrazuje parametry párové strategie s nejlepšími výsledky z pohledu RR
ratio. Tyto parametry jsme získali z IS optimalizace v aplikaci CSAnalyzer. V této tabulce jsou
zobrazeny jak páry z ETF trhů tak i páry z Futures trhů. Záměrně se páry mezi sebou
neprolínají, jelikož není správné vytvářet párové trhy mezi různými typy aktiv. Je tedy vhodné
tvořit páry pouze pro trhy ze stejných kategorií.

```
Tabulka 5: Vybrané parametry párové strategie z IS optimalizace na ETF a Futures
Trhy Parametry
```

```
DIA_EWG MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
DIA_IWM MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
DIA_SPY MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
EWG_DIA MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
EWG_IWM MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
EWG_SPY MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
IWM_DIA MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
IWM_EWG MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
IWM_SPY MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
SPY_DIA MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
SPY_EWG MeanRatioPeriod : 20StdDevPeriod : 12^
```

```
SPY_IWM MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
ZG_ZI MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
ZI_ZG MeanRatioPeriod : 20StdDevPeriod : 10^
```

```
Zdroj: [14], Vlastní zpracování
```

Podle této tabulky vytvoříme následující exekuční plány pro párové strategie aplikované na
ETF a Futures trhy v CSAnalyzer. Po vyběr správného databázového souboru s daty, lze tyto
exekuční plány vložit přímo do CSAnalyzeru a spustit testy. UC sloužící ke spustění OS
backtestu je v tomto případě UC09.

$-etf_oos_CSPS ETF | DIA_EWG | 1H | CSPS_Ratio_2d_0,20,10,500,2 | etf_oos
ETF | DIA_IWM | 1H | CSPS_Ratio_2d_0,20,10,500,2 | ETF | DIA_SPY | 1H | CSPS_Ratio_2d_0,20,10,500,2 | etf_oos etf_oos
ETF | EWG_DIA | 1H | CSPS_Ratio_2d_0,20,10,500,2 | ETF | EWG_IWM | 1H | CSPS_Ratio_2d_0,20,10,500,2 | etf_oos etf_oos
ETF | EWG_SPY | 1H | CSPS_Ratio_2d_0,20,10,500,2 | ETF | IWM_DIA | 1H | CSPS_Ratio_2d_0,20,10,500,2 | etf_oos etf_oos
ETF | IWM_EWG | 1H | CSPS_Ratio_2d_0,20,10,500,2 | ETF | IWM_SPY | 1H | CSPS_Ratio_2d_0,20,10,500,2 | etf_oos etf_oos
ETF | SPY_DIA | 1H | CSPS_Ratio_2d_0,20,10,500,2 | ETF | SPY_EWG | 1H | CSPS_Ratio_2d_0,20,10,500,2 | etf_oos etf_oos
ETF | SPY_IWM | 1H | CSPS_Ratio_2d_0,20,10,500,2 | ETF | SPY_EWG | 1H | CSPS_Ratio_2d_0,20,12,500,2 | etf_oos etf_oos

$-futures*oos_CSPS Future | @ZG#*@ZI# | 1H | CSPS*Ratio_2d_0,20,10,500,2 | futures_oos
Future | @ZI#*@ZG# | 1H | CSPS_Ratio_2d_0,20,10,500,2 | futures_oos

Díky postupům uvedeným v kapitole 3.3 Aplikace a jejích podkapitolách jsme byli schopni
realizovat out-of-sample testování. Toto testování nám umožnilo optimalizovat námi
definované strategie na vybraních trzích a následně aplikovat historickou simulaci na data,
které nebyli obsaženy v optimalizační fázi. Vše probíhalo za použití námi vytvořené aplikace
CSAnalyzer.

## 4. Zhodnocení výsledků

Stejně, jako jsme stanovili kroky pro správnou porovnatelnost napříč trhy a strategiemi v
kapitole 3.3 Aplikace, musíme i nyní stanovit podmínky, podle kterých budeme jednotlivé
strategie aplikované na trhy doporučovat k reálnému obchodování. Jsme tedy schopni správně
porovnávat dané výsledky pomocí dolarové neutrálnosti, relativní míře porovnání a
anualizovaní dosaženého výnosu. Neboli budeme v každé strategii investovat stejnou částku
do obchodu, porovnání výnosů bude v procentech a bude se jednat o anualizovaný výnos.
První a základní podmínkou je, že velikost anualizovaného výnosu nejlepších parametrů z IS
optimalizace musí být větší než 0. Tento základní předpoklad vychází z faktu, že strategie
nemůže být zisková v budoucnu, jestliže nebyla zisková na historických datech. Tento fakt je
navíc umocněn tím, že se jedná o nejlepší možný výsledek dosažený optimalizací parametrů.
Nejedná se tedy o žádný náhodný test. Tabulka níže zobrazuje anualizované výnosy
jednotlivých strategií po optimalizaci parametrů na vybraných trzích.

```
Tabulka 6: Podmínka 1 pro doporučení na reálné obchodování
```

```
Trh Strategie s parametry Anualizovaný výnos z IS Vyhovuje podmínce
```

```
SPY Buy and Hold,1000^ - 1,4 %^ NE^
DIA Buy and Hold,1000^ - 0,9 %^ NE^
IWM Buy and Hold,1000^ - 0,7 %^ NE^
EWG Buy and Hold,1000^ - 3,1 %^ NE^
ZG Buy and Hold,1000^ 4,1 %^ ANO^
ZI Buy and Hold,1000^ 9,8 %^ ANO^
SPY CSMS_SMAcrossSystem,10,50,1000,2^ - 2,3 %^ NE^
DIA CSMS_SMAcrossSystem,18,54,1000,2^ - 4,5 %^ NE^
IWM CSMS_SMAcrossSystem,30,54,1000,2^ 3,4 %^ ANO^
EWG CSMS_SMAcrossSystem,30,58,1000,2^ 8,4 %^ ANO^
ZG CSMS_SMAcrossSystem,30,70,1000,2^ 4,8 %^ ANO^
ZI CSMS_SMAcrossSystem,14,52,1000,2^ 10,3 %^ ANO^
DIA_EWG CSPS_Ratio_2d_0,20,10,500,2^ 16,8 %^ ANO^
```

```
DIA_IWM CSPS_Ratio_2d_0,20,10,500,2^ 13,2 %^ ANO^
DIA_SPY CSPS_Ratio_2d_0,20,10,500,2^ 9,6 %^ ANO^
EWG_DIA CSPS_Ratio_2d_0,20,10,500,2^ 16,9 %^ ANO^
EWG_IWM CSPS_Ratio_2d_0,20,10,500,2^ 23,6 %^ ANO^
EWG_SPY CSPS_Ratio_2d_0,20,10,500,2^ 22,9 %^ ANO^
IWM_DIA CSPS_Ratio_2d_0,20,10,500,2^ 13 %^ ANO^
IWM_EWG CSPS_Ratio_2d_0,20,10,500,2^ 24,3 %^ ANO^
IWM_SPY CSPS_Ratio_2d_0,20,10,500,2^ 17,1 %^ ANO^
SPY_DIA CSPS_Ratio_2d_0,20,10,500,2^ 9,6 %^ ANO^
SPY_EWG CSPS_Ratio_2d_0,20,12,500,2^ 22,9 %^ ANO^
SPY_IWM CSPS_Ratio_2d_0,20,10,500,2^ 17,2 %^ ANO^
ZG_ZI CSPS_Ratio_2d_0,20,10,500,2^ 7,6 %^ ANO^
ZI_ZG CSPS_Ratio_2d_0,20,10,500,2 6,4 % ANO
```

```
Zdroj: [14], Vlastní zpracování
```

Jak lze vidět v tabulkce výše, první podmínce vyhovuje většina IS optimalizací až na trhy SPY
a DIA na které jsme aplikovali tržní strategii SMAcrossover. Další podmínka souvisí s
odstraněním přeoptimalizovaných strategií. Tato podmínka definuje přeoptimalizovanost jako
stav, kdy OOS anualizovaný výnos nedosáhne ani 60 % anualizovaného výnosu z IS. Tuto
podmínku zastupuje tabulka níže.

```
Tabulka 7: Podmínka 2 pro doporučení na reálné obchodování
```

```
Trh Strategie s parametry
```

```
Anualizovaný výnos
z IS
```

```
Anualizovaný
výnos z OOS
```

```
Vyhovuje
podmínce
ZG Buy and Hold,1000^ 4,1 %^ 17,5 %^ ANO^
ZI Buy and Hold,1000^ 9,8 %^ 1,2 %^ NE^
```

```
IWM
```

```
CSMS_SMAcrossSystem,
30,54,1000,2 3,4 % 2,1 % ANO
```

```
EWG
```

```
CSMS_SMAcrossSystem,
30,58,1000,2 8,4 % -0,5 % NE
```

```
ZG
```

```
CSMS_SMAcrossSystem,
30,70,1000,2 4,8 % 9,5 % ANO
```

```
ZI
```

```
CSMS_SMAcrossSystem,
14,52,1000,2 10,3 % 15 % ANO
```

```
DIA_EWG
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 16,8 % 12 % ANO
```

```
DIA_IWM
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 13,2 % 3 % NE
```

```
DIA_SPY
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 9,6 % 7,2 % ANO
```

```
EWG_DIA
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 16,9 % 11,5 % ANO
```

```
EWG_IWM
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 23,6 % 7,9 % NE
```

```
EWG_SPY
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 22,9 % 11,2 % NE
```

```
IWM_DIA
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 13 % 3,1 % NE
```

```
IWM_EWG
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 24,3 % 6,3 % NE
```

```
IWM_SPY
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 17,1 % 4,2 % NE
```

```
SPY_DIA
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 9,6 % 7 % ANO
```

```
SPY_EWG
```

```
CSPS_Ratio_2d_0,20,12,5
00,2 22,9 % 9,9 % NE
```

```
SPY_IWM
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 17,2 % 4,6 % NE
```

```
ZG_ZI
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 7,6 % -14 % NE
```

```
ZI_ZG
```

```
CSPS_Ratio_2d_0,20,10,5
00,2 6,4 % -13,7 % NE
```

```
Zdroj: [14], Vlastní zpracování
```

Z tabulky výše je patrné, že podmínkou přeoptimalizace prošly jen některé strategie. Díky této
podmínce bychom měli se strategiemi dosahovat podobných výsledků jako v OOS i v reálném
obchodování. Další podmínkou je nutnost regresního koeficientu většího než 0. Tím

zaručíme, že se závislá proměnná (neboli kumulovaný výnos) změní o kladné číslo, pokud se
změní nezávislá proměnná (neboli čas) o jednotku. Díky této podmínce můžeme předpokládat
kladný budoucí výnos, jelikož nám regresní analýza udává kladný regresní koeficient. Poslední
podmínka je existence RR ratia většího nebo rovno než 0.3. Touto podmínkou zajistíme, že
budeme brát v potaz pouze takové výsledky, které svým výnosem budou dosahovat alespoň
3/10 podstupovaného rizika. Ideální případ nastává v případě, že výnos je větší než
podstupované riziko, čili RR ratio je větší než jedna. Těchto případů se ale v praxi dosahuje
zřídka s tímto způsobem výpočtu. Tabulka níže zobrazuje tyto dvě podmínky.

```
Tabulka 8: Podmínka 3 a 4 pro doporučení na reálné obchodování
```

```
Trh Strategie s parametry koeficient Regresní RR ratio Vyhovuje podmínce
```

```
ZG Buy and Hold,1000^ 0,005 %^ 0,06^ NE^
IWM CSMS_SMAcrossSystem,30,54,1000,2^ 0,1 %^ 0,2^ NE^
ZG CSMS_SMAcrossSystem,30,70,1000,2^ 0,5 %^ 0,5^ ANO^
ZI CSMS_SMAcrossSystem,14,52,1000,2^ 0,6 %^ 0,5^ ANO^
DIA_EWG CSPS_Ratio_2d_0,20,10,500,2^ 0,2 %^ 0,3^ ANO^
DIA_SPY CSPS_Ratio_2d_0,20,10,500,2^ 0,06 %^ 0,2^ NE^
EWG_DIA CSPS_Ratio_2d_0,20,10,500,2^ 0,2 %^ 0,3^ ANO^
SPY_DIA CSPS_Ratio_2d_0,20,10,500,2^ 0,06 %^ 0,2^ NE^
```

```
Zdroj: [14], Vlastní zpracování
```

Z výše uvedené tabulky vyplývá, že našimi podmínkami prošly 2 tržní strategie a 2 párové
strategie. Strategie **Buy and Hold** se neukázala jako vhodná pro námi zvolené trhy. Naopak
**SMAcrossover** strategie vyhovuje pro reálné obchodování na Futures trzích ZG a ZI, kde
dosáhla OOS ročního výnosu 9,5 % a 15 %. Vzhledem k RR ratio, jež je 0,5 v obou
případech, můžeme považovat poměr výnosu a podstupovaného rizika za dostatečný k
doporučení na reálné obchodování. Jinými slovy by se dalo říci, že za výnos 5 %
podstupujeme riziko o velikosti 10 %. Strategie **Ratio_2d_0** naopak vyhovovala na ETF
trzích, kde byly vybrány párové trhy DIA_EWG a EWG_DIA. Na těchto trzích dosáhla OOS
ročního výnosu 12 % a 11,5 %, v poměru k RR ratio, jež je 0,3 v obou případech. I takové RR
ratio lze považovat za vhodné k reálnému obchodování.

## 5. Závěr.........................................................................................................................................

Cílem této bakalářské práce bylo popsat, implementovat, aplikovat a porovnat tři investiční
strategie na vybraná aktiva. V rámci práce byla představena teoretická rovina pro tvorbu a
aplikaci investičních strategií, která byla následně prakticky analyzována, navrhnuta a
implementována do aplikace CSAnalyzer. Implementace probíhala pomocí objektově
orientovaného programovacího jazyka C# a .NET Frameworku 4.0. Pro splnění stanovených
požadavků na aplikaci, bylo nutné zajistit komunikaci s poskytovatelem dat a statistickým
softwarem. Jako poskytovatel dat byl vybrán IQFeed, který po registraci poskytuje kvalitní
OHLC data. Za statistický software byl zvolen Wolfram Mathematica 8.0 a komunikace s
.NET Frameworkem probíhala pomocí Wofram.NETLink.

Po implementaci byl využit CSAnalyzer k provedení out-of-sample testů. Z vybraných tří
investičních strategií nebyla jako jediná doporučena k reálnému obchodování strategie Buy and
Hold. Této strategii se nepodařilo realizovat uspokojivé výsledky ani na jednom z vybraných
aktiv. Strategie SMAcrossover dosáhla uspokojivých výsledků pouze na komoditních aktivech.
Těmi byly zvoleny ZG a ZI, neboli trhy zlata a stříbra. Na těchto trzích dosáhla výnosu 9,5 %
a 15% s poměrem rizika a výnosu 0,5 v obou případech. Párová strategie Ratio_2d_0 pak
dopadla nejlépe na trzích ETF. Těmi byly zvoleny trhy SPY,DIA,EWG a IWM, neboli
zástupci hlavních světových indexů spolu s německým indexem. Na těchto trzích dosáhla
nejuspokojivějšího výsledku na páru složeném z aktiv DIA a EWG, kde dosáhla výnosu 12 %
s poměrem rizika a výnosu 0,3.

CSAnalyzer aplikace umožňuje uživatelům vytváření vlastních testů, jejich vyhodnocování,
ukládání do úložistě a stahování dat. Aplikace může najít své uplatnění především pro
programátory, kteří mají zájem o testování svých investičních strategií. Jelikož aplikace splňuje
požadavek snadné rozšiřitelnosti, lze dopsat do řešení vlastní strategii a testovat její výkonnost
podle stanovených postupů.

V rámci této bakalářské práce došlo k naplnění všech stanovených cílů a požadavků,
které na ni byly kladeny. Pro výhled do budoucna bych doporučil výměnu úložiště realizované
serializací za relační databázi. Spolu s použitím money managementu by se mohlo jednat o
úpravy vedoucí ke kvalitnější testovací platformě a lepším dosáhnutým výsledkům.

The main goal of this thesis work was to describe, implement, apply and compare three
investment strategies on selected assets. In one of the thesis topic the author has introduced
theoretical level for creation and application of investment strategies. These strategies were
practically analyzed afterward. After proper analysis these strategies were well designed and
implemented in CSAnalyzer application. For implementation was selected object oriented
language C# and .NET framework 4.0. To ensure fulfilling of requirements given by the
application it was required to establish a communication flow between data provider and static
software. As data provider the author of this thesis selected IQFeed, which after proper
registration ensure high quality data. As static software was selected Wolfram Mathematica 8.0
and the communication with .NET framework and static software was serviced by
Wolfram.NETLink.

For the implementation and out-of-sample test CSAnalyzer was used. From three investment
strategies was not recommended for real trading only strategy Buy and Hold. This strategy was
not able to realize satisfactory results on any of selected assets. The SMAcrossover strategy
reached pretty agreeable outcomes only on commodity assets. Author of this thesis selected
ZG and ZI assets; we refer them as markets of gold and silver. On this markets the strategy
achieved revenue of 9,5% and 15% with ratio of risk and revenue of 0.5 in both cases. The
pair strategy Ratio_2d_0 went the best on ETF markets. SPY, DIA, EWG and IWM was
selected in this comparison as they are representatives of all world indexes and German index.
On these markets the strategy reached the best results on assets from DIA and EWG, where
the strategy got revenue of 12% with ratio of risk and revenue of 0.3.

CSAnalyzer application allows users to create their own tests, test evaluation, persisting and
downloading of data. This application best use is for programmers who would like to test
investment strategies. Since this application has very good modularity programmers can easily
add their own strategy and test it according specified procedures.

In this thesis work the author has achieved all specified goals and requirements that were
conducted at the beginning of this work. For the future use the author would recommend
switching of persistence realized by serializing data to relation database. Together with using
of money management these changes could be marked as major for accomplishing better test
platform and improved results.

## 6. Seznam použitých zdrojů...........................................................................................................

1. MIKULA, Štěpán. _Technická analýza_. Brno, 2006. Seminární práce. Masarykova
   univerzita
2. FAMA, Eugen. _Efficient Capital Markets: A Review of Theory and Empirical Work, Journal of_
   _Finance, Volume 25, Issue 2_ , Papers and Proceedings of the Twenty-Eighth Annual
   Meeting of the American Finance Association New York, N.Y. December, 28-30,
   1969 (May, 1970), 383-417
3. EHRMAN, Douglas S. _The Handbook of Pairs Trading_ , Wiley (2006); ISBN-13 978-0-
   471-72707-1
4. GRECH, Anthony. _Equity pairs: a Trading Strategy_ , IG Index (2009);
5. Unicorn Universe (UCL-BT:STA12S.CZ/LEC09/GL)
6. PARDO, Robert. _The Evaluation and Optimization of Trading Strategies_ , Wiley (2008);
   ISBN 978-0-470-12801-5
7. ARONSON, David. _Evidence-based technical analysis, Applying the Scientific Method and_
   _Statistical Inference to Trading Signals_ , Wiley (2007); ISBN 0-470-00874-1
8. MANGANO, Sal. _Mathematica Cookbook_ , O’Reilly (2010); ISBN 978-0-596-52099-1
9. BALEJ, Marek. _Síťové protokoly a sockety_. České Budějovice, 2010. Bakalářská práce
   (Bc.). Jihočeská univerzita v Českých Budějovicích
10. VESELÁ, Jitka. _Investování na kapitálových trzích v příkladech_ , Oeconomica (2007); ISBN
    978-80-2451-166-5
11. LANDOROVÁ, Anděla. _Cenné papíry a finanční trh_. Liberec: Technická univerzita v
    Liberci, 2005. ISBN 80-7083-920-1
12. CHANDE, Tushar S. _Beyond Technical Analysis : How to Develop and Implement a Winning_
    _Trading System_ , Wiley (1997); ISBN 0-471-16188-8
13. Unicorn Universe (UCL-BT:STA12S.CZ/LEC08/GL)
14. CSAnalyzer [program na DVD]. 2012, Praha. Kellerstein Lukáš
15. StockCharts.com, ChartSchool. [cit. 2012-01-05]. Dostupné z URL :
    <http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mo
    ving_averages>
16. VIDYAMURTHY, Ganapathy. _Pairs Trading : Quantitative Methods and Analysis_ , Wiley
    (2004); ISBN 0-471-46067-2
17. JACOBS, Bruce I. _Market Neutral Strategies_ , Wiley (2005); ISBN 0-471-26868-2

18. NICHOLAS, Joseph G. _Long/Short Hedge Fund Strategies_ , Bloomberg press (2000);
    ISBN 1-57660-037-8
19. Wolfram Mathematica 8, Documentation center. [cit. 2012-02-10]. Dostupné z URL :
    <http://reference.wolfram.com/mathematica/guide/Mathematica.html>
20. Wolfram Mathematica 8, .NETLink namespace. [cit. 2012-02-10]. Dostupné z URL :
    <http://reference.wolfram.com/mathematica/NETLink/ref/net/Wolfram.NETLink
    .html>
21. MSDN knihovna, Microsoft. [cit. 2012-02-15]. Dostupné z URL :
    <http://msdn.microsoft.com/cs-cz/library>
22. Poznáváme C# a Microsoft.NET, 40.díl - Serializace, Praha: Petr Puš [cit. 2011-11-
    20]. Dostupné z URL : <http://www.zive.cz/Clanky/Poznavame-C-a-
    MicrosoftNET-40-dil--serializace/sc-3-a-126553/default.aspx>
23. Unicorn Universe (UCL-BT:TFM12S.CZ/LEC07/GL)
24. Unicorn Universe (UCL-BT:TFM12S.CZ/LEC05/GL)

## 7. Seznam zkratek

```
Zkratka Popisek
IT Informační technologie
SMA Simple moving average
IS In-sample
OOS Out-of-sample
WFE Walk-forward efficiency
SE Standard Errors
VaRisk Value at Risk
VaReturn Value at Return
GUI Graphical User Interface
OHLC Open, High, Low, Close
WPF Windows Presentation Foundation
```

## 8. Seznam obrázků

Obrázek 1: Grafické znázornění jednoduchého klouzavého průměru ......................................... 16
Obrázek 2: Základní pravidla SMA crossover graficky .................................................................. 17
Obrázek 3: Grafické znázornění indikárů pro párovou strategii ................................................... 20
Obrázek 4: Grafické znázornění indikárů pro párovou strategii v normalizovaném tvaru ....... 21
Obrázek 5: Základní pravidla párové strategie graficky .................................................................. 22
Obrázek 6: Grafické znázornění optimalizace dvou vstupních parametrů .................................. 24
Obrázek 7: In-sample optimalizace .................................................................................................... 25
Obrázek 8: Out-of-sample backtest ................................................................................................... 25
Obrázek 9: Magický trojúhelník investora ........................................................................................ 28
Obrázek 10: Pravděpodobnost kladného výnosu ............................................................................ 30
Obrázek 11: Hustota pravděpodobnosti s VaRisk a VaReturn ..................................................... 31
Obrázek 12: 3-vrstvá architektura ...................................................................................................... 39
Obrázek 13: Popis komunikace s CSAnalyzer ................................................................................. 40
Obrázek 14: Rozdělení aplikace do vrstev a logických entit .......................................................... 41
Obrázek 15: Prezentační vrstva a interakce s okolím ...................................................................... 43
Obrázek 16: DataManager package ................................................................................................... 44
Obrázek 17: DataAnalyzer package ................................................................................................... 44
Obrázek 18: DataViewer package ...................................................................................................... 45
Obrázek 19: Formulář backtesty tržních strategií ............................................................................ 46
Obrázek 20: Detailní formulář backtestu tržní strategie ................................................................. 47
Obrázek 21 : Aplikační vrstva a interakce s okolím ........................................................................ 48
Obrázek 22: Implementace trhů v CSAnalyzer ................................................................................ 49
Obrázek 23: Strategy pattern investičních analýz v CSAnalyzer ................................................... 50
Obrázek 24: Vstup a výstup investiční strategie v CSAnalyzer ..................................................... 51
Obrázek 25: Implementace backtestu v CSAnalyzer ...................................................................... 52
Obrázek 26: Sekvenční diagram backtestu v CSAnalyzer .............................................................. 53
Obrázek 27: Implementace optimalizace v CSAnalyzer ................................................................. 54
Obrázek 28: Sekvenční diagram optimalizace v CSAnalyzer ......................................................... 55
Obrázek 29: Sekvenční diagram stahování dat v CSAnalyzer ........................................................ 57
Obrázek 30: Sekvenční diagram volání existující metody v Wolfram Mathematica 8.0 ............ 58
Obrázek 31: Perzistentní vrstva a interakce s okolím ..................................................................... 59
Obrázek 32: Implementace datové vrstvy v CSAnalyzer ............................................................... 60
Obrázek 33: Parametry tržní strategie v CSAnalyzer ...................................................................... 65
Obrázek 34: Parametry optimalizace v CSAnalyzer ........................................................................ 65
Obrázek 35: Parametry tržní strategie v CSAnalyzer ...................................................................... 66
Obrázek 36: Parametry optimalizace v CSAnalyzer ........................................................................ 67

## 9. Seznam tabulek

Tabulka 1: Seznam a popis případů užití .......................................................................................... 34
Tabulka 2: Vybrané trhy ETF a Futures ........................................................................................... 62
Tabulka 3: Vybrané párové trhy ETF a Futures .............................................................................. 63
Tabulka 4: Vybrané parametry tržní strategie z IS optimalizace na ETF a Futures ................... 68
Tabulka 5: Vybrané parametry párové strategie z IS optimalizace na ETF a Futures ............... 69
Tabulka 6: Podmínka 1 pro doporučení na reálné obchodování .................................................. 71
Tabulka 7: Podmínka 2 pro doporučení na reálné obchodování .................................................. 72
Tabulka 8: Podmínka 3 a 4 pro doporučení na reálné obchodování ............................................ 74

## 10. Seznam Příloh

# 1. DVD obsahující text práce, implementovanou aplikaci a zdrojové kódy

### 10.1 DVD obsahující text práce, implementovanou aplikaci a zdrojové kódy

DVD je přiloženo do vazby tištěné verze bakalářské práce.

Struktura DVD:

- Složka Bakalářská práce
  o .docx soubor s textem bakalářské práce: Kellerstein Lukas - Bakalarska
  prace.docx
  o .pdf soubor s textem bakalářské práce: Kellerstein Lukas - Bakalarska
  prace.pdf
- Složka CSAnalyzer
  o CSAnalyzerCore.dll
  o CSAnalyzerWinForm.vshost.exe
  o CSAnalyzerWinForm.vshost.exe.manifest
  o CSDataAnalyzer.dll
  o CSDataManager.dll
  o CSSerializer.dll
  o CSWolframMathematica.dll
  o Mono.Reflection.dll
  o Wolfram.NETLink.dll
- Složka CSAnalyzer_zdrojove_kody
  o Obsahuje soubory jednotlivých tříd popsaných v této práci včetně
  projektu vytvořeného v MS Visual Studio 2010 Ultimate
- CSAnalyzer_KellersteinZBP
  o Ciselniky
   ETFs_excel.csv
   Futures_excel.csv
  o CSAnalyzerWolframWorkspace
   .metadata
   CSAnalyzerWolframProject
  o Db
   etf_is
   etf_is_CSMS
   etf_is_CSPS
   etf_oos
   etf_oos_CSMS
   etf_oos_CSPS
   futures_is
   futures_is_CSMS
   futures_is_CSPS
   futures_oos
   futures_oos_CSMS
   futures_oos_CSPS
