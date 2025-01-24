# KAi_Mottakskontroll_segmentering
Tekstlig beskrivelse: Teste KartAI-algoritmer på utvalgt område i et GeoVekst-prosjekt, for å sjekke om KartAI-algoritmene kan detektere bygninger på et nivå som kan være til hjelp ved mottakskontroll av bygningsdata konstruert fra flybilder.  


## Generell info

Denne koden bruker PyTorch sitt bibliotek for maskinlæring og utvidelsen TorchGeo for geografisk informasjon til å trene opp og kjøre prediksjoner med modellen FarSeg. FarSeg - Foreground-Aware Relation Segmentation Network - er en segmenterings-modell som er spesialisert på å fokusere på framgrunnen i bildet. I de fleste bilder er det stort sett bakgrunnen som tar opp det meste av innholdet og dermed er styrende for hva maskinlæringsmodeller fanger opp og tar tak i. FarSeg derimot bryter opp bildet i flere lag, og vektlegger de ulike lagene på en måte som nøytralisere overvekten av informasjon fra bakgrunnen. Modellen lærer seg også relasjonene mellom bakgrunnen og de elementene den skal segmentere. På den måten kan områdene som dekkes av de ulike objekttypene en er interessert i segmenteres helt og sammenlignes.

Formålet med dette prosjektet er å undersøke om denne modellen kan segmentere ut objektene godt nok til å utføre delene av mottakskontroll som omfatter geometrisk nøyaktighet. Dette vil si å sjekke hvor god posisjonering og detaljeringsgrad de manuelt konstruerte dataene har.

Koden her utfører semantisk segmentering, både trening av modellen og utførelsen av prediksjoner i etterkant ved å laste opp en trent modell. Under dette prosjektet er koden designet for bygninger og veier, men den kan også trenes på andre og flere objekttyper. Det er også inkludert kode for å lage maskeringer og statistikk over grunndataene, og validering av de predikerte resultatene.

Om en ønsker å vite mer om FarSeg-modellen, kan en lese teksten til Zhuo Zheng, Yanfei Zhong, Junjue Wang og Ailong Mai her:

https://arxiv.org/pdf/2011.09766

# Komme i gang med prosjektet

Klon prosjektet og generer et nytt virtuelt miljø i kodebasen. For å komme i gang med å bruke programmet følger en de påfølgende seksjonene som tar for seg hvordan en får koden til å fungere og hva de ulike filene gjør.

## Laste ned pakker

Først må en laste ned hovedbibliotekene til PyTorch fra deres nettside (https://pytorch.org/). Koden generert i dette prosjektet bruker torch 2.5.1 med kompabilitet for NVIDIA sin CUDA 12.0 GPU. For å få inn riktig torch-bibliotek brukt i denne koden, kjør:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Når dette er gjort kan resten av pakkene lastes ned ved å kjøre:

pip install -r requirements.txt

## Programmet

Dette programmet består av en fil *main.py* som kaller alle de andre aktuelle funksjonene. Alternativene i dette programmet er som følger:

1. Trening av en ny FarSeg modell
2. Predikering ved hjelp av en trent FarSeg modell
3. Validering av utførte prediksjoner
4. Generering av statistikk basert på GeoTIFFs og Geopackages

Hver del kan kjøres separat, men trinn 2 og 3 krever at henholdsvis trinn 1 og 2 er kjørt minst én gang tidligere. *main.py* gir en oversikt over disse, og brukeren gir input om hva en ønsker. Under følger en mer detaljert beskrivelse av hva hver del gjør.

### 1. Trening

Trening av en ny modell utføres i *mainTrain.py* under *Program*. For å kjøre denne koden skriver en '1' som input i *main.py*.

På lik linje som i *main.py* må brukeren gi inn input i starten av programmet for å finne rett data. Her må en først gi inn:

- *geopackage_folder*: Filsti til mappen med dine lagrede geopackages
- *geotiff_folder*: Filsti til mappen med dine lagrede GeoTIFFs for trening
- *tile_folder*: Filsti til en ny mappe som oprettes midlertidig for å lagre midlertidige tiles av GeoTIFFs

*geopackage_folder* inneholder dine geografiske data som vil fungere som fasit. Det skal være én fil for hver objekttype. **For koden generert her må geopackage-filene ligge i riktig rekkefølge - bygg først, så veier.**

Neste steg er å gi inn antall *batches*, *epochs* og *workers* som en vil bruke til treningen.

Avslutningsvis gir en inn stien til en ny mappe (*model_path*) hvor en vil lagre den trente modellen. Deretter skriver en inn navnet (*model_name*) på modellen en skal trene. *model_name* må ende på '.pth'.

Funksjonen vil deretter automatisk opprette en ny pre-trained FarSeg-modell og trene på alle GeoTIFFs som har aktuell data i geopackages. Tiles opprettes og slettes i mappen *tile_folder*. Når modellen er ferdig trent vil den lagre modellen som *model_path + model_name*, og slette *tile_folder*.

### 2. Prediksjoner

Når en har en trent modell, kan en kjøre prediksjoner med denne i *mainInference.py* i *Program* ved å skrive inn '2' som input i *main.py*.

Som med treningen, må en også her spesifisere hvor data hentes fra. Her gir en inn følgende input:

- *model_path*: Filsti til den trente modellen din
- *geotiff_folder*: Filsti til mappen med dine lagrede GeoTIFFs for predikering
- *tile_folder*: Filsti til en ny mappe som opprettes midlertidig for å lagre midlertidige tiles av GeoTIFFs
- *segmented_folder*: Filsti til en ny mappe som opprettes midlertidig for å lagre midlertidige segmenterte tiles av GeoTIFFs
- *output_folder*: Filsti til en ny mappe der de endelige resultatene av prediksjonen skal lagres

Programmet itererer over alle GeoTIFF-filene i *geotiff_folder* og ufører prediksjonen og lagrer sluttresultatene i *output_folder*. *tile_folder* og *segmente_folder* slettes avslutningsvis.

### 3. Validering

Når en har kjørt prediksjoner med en trent FarSeg-modell kan en validere de genererte resultatene med *mainValidation.py* under *Program* ved å gi inn '3' i *main.py*.

Her gir en inn følgende:

- *result_folder*: Filsti til mappen med prediksjoner / segmenterte bilder
- *geopackage_folder*: Filsti til mappen der aktuelle geopackages er lagret
- *mask_folder*: Filsti til en ny midlertidig mappe som lagrer midlertidige maskeringer av GeoTIFF-filene
- *log_file*: Filsti til en ny log-fil som lagrer alle IoU-resultater

**Pass på at filene i *geopackage_folder* ligger i riktig rekkefølge - bygninger og så vei.**

Programmet iterere over alle de segmenterte bildene i *result_folder*, lager tilhørende maskering fra geopackages og skriver kvaliteten på prediksjonen i form av Intersection over Union - IoU - til log-filen.

### 4. Statistikk

Hvis en ønsker statistikk om GeoTIFF-filene en har, kan en gi inn '4' i *main.py* som vil koble seg til *dataStatistics.py* i *Functionality*.

Her gir en inn:

- *geopackages*: Filsti til aktuelle geopackages
- *geotiff_folder*: Filsti til mappen der en har lagret GeoTIFFs
- *tile_folder*: Filsti til en ny midlertidig mappe som lagrer midlertidige tiles av GeoTIFFs

Denne koden vil telle totalt antall tiles av GeoTIFF-filene, hvor mange av disse som er gyldige og ugyldige (ugyldig = består kun av *nodata value*), hvor mange tiles med bygninger, antall tiles med veier, antall tiles med x antall bygninger, antall tiles med x antall veier, og til slutt lage et plott for hver av disse to siste tilfellene. Det vil si at x-aksen er antall bygninger / veier i tilen, og y-aksen er antall slike tiles. På den måten vil en kunne se hvor mange tiles som f.eks. har fem bygninger eller tre veier.

Dette vil gi en grei oversikt over datasettet en har tilgjengelig og hvor mange tiles som vil bli brukt til trening.

### Ellers

*For ytterligere instruksjoner henvises det til kommentarene i Python-filene.*
