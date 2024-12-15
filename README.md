# KAi_Mottakskontroll_segmentering
Tekstlig beskrivelse: Teste KartAI-algoritmer på utvalgt område i et GeoVekst-prosjekt, for å sjekke om KartAI-algoritmene kan detektere bygninger på et nivå som kan være til hjelp ved mottakskontroll av bygningsdata konstruert fra flybilder.  


## Generell info

Denne koden bruker PyTorch sitt bibliotek for maskinlæring og utvidelsen TorchGeo for geografisk informasjon til å trene opp og kjøre prediksjoner med modellen FarSeg. FarSeg - Forground-Aware Relation Segmentation Network - er en segmenterings-modell som er spesialisert på å fokusere på framgrunnen i bildet. I de fleste bilder er det stort sett bakgrunnen som tar opp det meste av innholdet og dermed er styrende for hva maskinlæringsmodeller fanger opp og tar tak i. FarSeg derimot bryter opp bildet i flere lag, og vektlegger de ulike lagene på en måte som nøytralisere overvekten av informasjon fra bakgrunnen. Modellen lærer seg også relasjonene mellom bakgrunnen og de elementene den skal segmentere. På den måten kan områdene som dekkes av de ulike objekttypene en er interessert i segmenteres helt og sammenlignes.

Formålet med dette prosjektet er å undersøke om denne modellen kan segmentere ut objektene godt nok til å utføre delene av mottakskontroll som omfatter geometrisk nøyaktighet. Dette vil si å sjekke hvor god posisjonering og detaljeringsgrad de manuelt konstruerte dataene har.

Koden her utfører semantisk segmentering, både trening av modellen og utførelsen av prediksjoner i etterkant ved å laste opp en trent modell. Under dette prosjektet er koden designet for bygg og veier, men den kan også trenes på andre og flere objekttyper. Det er også inkludert kode for å lage maskeringer og statistikk over grunndataene, og validering av de predikerte resultatene.

Om en ønsker å vite mer om FarSeg-modellen, kan en lese teksten til Zhuo Zheng, Yanfei Zhong, Junjue Wang og Ailong Mai her:

https://arxiv.org/pdf/2011.09766

# Komme i gang med prosjektet

Klon prosjektet og generer et nytt virtuelt miljø i kodebasen. For å komme i gang med å bruke programmet følger en de påfølgende seksjonene som tar for seg hvordan en får koden til å fungere og hva de ulike filene gjør.

## Laste ned pakker

Først må en laste ned hovedbibliotekene til PyTorch fra deres nettside (https://pytorch.org/). Koden generert i dette prosjektet bruker torch 2.5.1 med kompabilitet for NVIDIA sin CUDA 12.0 GPU. For å få inn riktig torch-bibliotek brukt i denne koden, kjør:

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Når dette er gjort kan resten av pakkene lastes ned ved å kjøre:

pip install -r requirements.txt

## Programmet

Programmet er delt inn i ulike deler, der hver del kjøres separat uavhengig av hverandre:

- Trening
- Predikering
- Validering

Derimot krever de senere delene at en har kjørt de første minst en gang først.

### 1. Trening

Trening av en ny modell utføres i *main_train.py* under *Training*.

Her må en definere hvor treningsdataen er lagret før en setter i gang. Variablene *geopackage_folder* og *geotiff_folder* må endres slik at filstiene peker dit dataen er lagret. *geopackage_folder* skal inneholde alle geopackage-filene med dataen en skal trene på. Dette er vektor-data som fungerer som fasit i treningen. **For koden generert her må geopackage-filene ligge i riktig rekkefølge - bygg først, så veier.** I *geotiff_folder* legger en alle GeoTIFF-bildene en skal trene på.

En må også endre siste del av *model_name* slik at den trente modellen får et navn du kjenner igjen.

Tiles den genererer og sletter underveis, i tillegg til mappen der modellen lagres, genereres underveis og trengs ikke tas hensyn til.

### 2. Prediksjoner

Når en har en trent modell, kan en kjøre prediksjoner med denne i *main_inference.py* eller *main_inference_ipynb* i mappen *Inference*.

Som med treningen, må en også her spesifisere hvor data hentes fra. Før koden kjøres må en gi inn filstien til modellen i *modelPath*. Videre må dataene som en skal predikere på (GeoTIFF-filer) spesifiseres ved å skrive mappestien i *ortophoto_path*.

Mappene der tiles genereres underveis og der sluttresultatet lagres opprettes automatisk. Det en derimot må sjekke er at filstiene til bildene som genereres til slutt, *output_original* og *output_segmented*, stemmer overens med mappen til sluttresultatene.

Koden vil segmentere alle GeoTIFF-filene i mappen en gir inn.

### 3. Validering

Når en har kjørt prediksjoner med en trent FarSeg-modell kan en validere de genererte resultatene med *validate.py* under *Validation*.

Her skal programmet automatisk hente segmenteringene fra mappen der prediksjonene lagres. I tillegg lages det en egen mappe som genererer maskeringer basert på geopackage-dataen. Her må en gi inn riktig filsti til mappen i *geopackage_folder*. **Pass på at filene ligger i riktig rekkefølge - bygg og så vei.**

Programmet genererer automatisk en loggfil som inneholder all nyttig informasjon om Intersection over Union - IoU - til hver av GeoTIFF-bildene.

### Annet

Om en ønsker generell informasjon om datasettet en har, kan en gå inn i *Data* der en har *data_statistics.py* og *data_statistics.ipynb*. Dette er en fil som printer følgende:

- Antall tiles i datasettet ditt
- Antall gyldige tiles
- Antall ugyldige tiles
- Antall tiles med byggninger
- Antall tiles med veier
- Fordeling av antall bygg per tile
- Fordeling av antall veier per tile

I tillegg plotter den fordelingen av antall bygg og veier per tile. Det vil si, hvor mange tile har ett bygg, to bygg, osv.

Her må en gi inn filstien til mappen der GeoTIFF-filene er lagret og filstiene til geopackage-lagene i henholdsvis *geotiff_folder*, *building_layer* og *road_layer*.

For å generere maskeringer, fasit, av GeoTIFF basert på geopackage-lagene kan *createMasks.py* benyttes (også i *Data*).

Her gir en inn hvor geopackage-lagene og GeoTIFF-filene er lagret i henholdsvis *geopackage_folder* og *geotiff_folder*.

### Ellers

*For ytterligere instruksjoner henvises det til kommentarene i Python-filene.*
