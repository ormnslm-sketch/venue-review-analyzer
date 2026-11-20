"""Prepare unverified Turkish and Spanish translations of verified EN sources."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MAMS_PATH = CURATION_DIR / "mams_neutral_candidates_en.csv"
HUMAN_PATH = CURATION_DIR / "human_authored_neutral_candidates_en.csv"
TR_PATH = CURATION_DIR / "translated_neutral_candidates_tr.csv"
ES_PATH = CURATION_DIR / "translated_neutral_candidates_es.csv"
TR_BATCH_PATH = CURATION_DIR / "review_batches" / "neutral_tr_translation_batch_01.csv"
ES_BATCH_PATH = CURATION_DIR / "review_batches" / "neutral_es_translation_batch_01.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "en_freeze_tr_es_translation_summary.txt"

FIELDS = [
    "candidate_id",
    "parallel_source_group_id",
    "english_source_id",
    "english_sentence",
    "translated_sentence",
    "language",
    "mapped_aspect",
    "proposed_sentiment",
    "original_source_type",
    "original_source_dataset",
    "original_source_license",
    "translation_human_verified",
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "review_notes",
]

SELECTED_IDS = {
    "taste": [
        "MAMS-EN-000165-FOOD",
        "MAMS-EN-000793-FOOD",
        "MAMS-EN-001004-FOOD",
        "MAMS-EN-001428-FOOD",
        "MAMS-EN-001950-FOOD",
        "MAMS-EN-001951-FOOD",
        "MAMS-EN-002173-FOOD",
        "MAMS-EN-002395-FOOD",
        "MAMS-EN-002497-FOOD",
        "MAMS-EN-002982-FOOD",
    ],
    "atmosphere": [
        "HA-CAND-EN-ATMOSPHERE-0001",
        "HA-CAND-EN-ATMOSPHERE-0004",
        "HA-CAND-EN-ATMOSPHERE-0007",
        "HA-CAND-EN-ATMOSPHERE-0010",
        "HA-CAND-EN-ATMOSPHERE-0013",
        "HA-CAND-EN-ATMOSPHERE-0016",
        "HA-CAND-EN-ATMOSPHERE-0019",
        "HA-CAND-EN-ATMOSPHERE-0022",
        "HA-CAND-EN-ATMOSPHERE-0029",
        "HA-CAND-EN-ATMOSPHERE-0030",
        "HA-CAND-EN-ATMOSPHERE-0031",
        "HA-CAND-EN-ATMOSPHERE-0032",
        "HA-CAND-EN-ATMOSPHERE-0033",
        "HA-CAND-EN-ATMOSPHERE-0034",
        "MAMS-EN-000544-AMBIENCE",
        "MAMS-EN-000556-AMBIENCE",
        "MAMS-EN-001960-AMBIENCE",
        "MAMS-EN-002555-AMBIENCE",
    ],
    "service": [
        "HA-CAND-EN-SERVICE-0001",
        "HA-CAND-EN-SERVICE-0005",
        "HA-CAND-EN-SERVICE-0009",
        "HA-CAND-EN-SERVICE-0013",
        "HA-CAND-EN-SERVICE-0017",
        "HA-CAND-EN-SERVICE-0021",
        "HA-CAND-EN-SERVICE-0024",
        "HA-CAND-EN-SERVICE-0028",
        "HA-CAND-EN-SERVICE-0043",
        "HA-CAND-EN-SERVICE-0045",
        "HA-CAND-EN-SERVICE-0046",
        "HA-CAND-EN-SERVICE-0049",
        "HA-CAND-EN-SERVICE-0052",
        "HA-CAND-EN-SERVICE-0054",
        "HA-CAND-EN-SERVICE-0055",
        "HA-CAND-EN-SERVICE-0059",
        "HA-CAND-EN-SERVICE-0060",
        "HA-CAND-EN-SERVICE-0061",
        "HA-CAND-EN-SERVICE-0062",
        "HA-CAND-EN-SERVICE-0063",
        "HA-CAND-EN-SERVICE-0064",
        "HA-CAND-EN-SERVICE-0065",
    ],
    "value": [
        "HA-CAND-EN-VALUE-0001",
        "HA-CAND-EN-VALUE-0005",
        "HA-CAND-EN-VALUE-0009",
        "HA-CAND-EN-VALUE-0017",
        "HA-CAND-EN-VALUE-0020",
        "HA-CAND-EN-VALUE-0039",
        "HA-CAND-EN-VALUE-0041",
        "HA-CAND-EN-VALUE-0043",
        "HA-CAND-EN-VALUE-0050",
        "HA-CAND-EN-VALUE-0051",
        "HA-CAND-EN-VALUE-0052",
        "HA-CAND-EN-VALUE-0053",
        "HA-CAND-EN-VALUE-0054",
        "HA-CAND-EN-VALUE-0055",
        "HA-CAND-EN-VALUE-0056",
        "HA-CAND-EN-VALUE-0057",
        "HA-CAND-EN-VALUE-0058",
        "HA-CAND-EN-VALUE-0059",
        "HA-CAND-EN-VALUE-0060",
    ],
}

REVIEW_SOURCE_IDS = [
    "MAMS-EN-001004-FOOD",
    "MAMS-EN-002395-FOOD",
    "MAMS-EN-002173-FOOD",
    "HA-CAND-EN-ATMOSPHERE-0001",
    "HA-CAND-EN-ATMOSPHERE-0004",
    "HA-CAND-EN-ATMOSPHERE-0007",
    "HA-CAND-EN-ATMOSPHERE-0013",
    "HA-CAND-EN-ATMOSPHERE-0019",
    "HA-CAND-EN-SERVICE-0001",
    "HA-CAND-EN-SERVICE-0009",
    "HA-CAND-EN-SERVICE-0013",
    "HA-CAND-EN-SERVICE-0017",
    "HA-CAND-EN-SERVICE-0024",
    "HA-CAND-EN-SERVICE-0028",
    "HA-CAND-EN-VALUE-0001",
    "HA-CAND-EN-VALUE-0005",
    "HA-CAND-EN-VALUE-0009",
    "HA-CAND-EN-VALUE-0017",
    "HA-CAND-EN-VALUE-0052",
]

# source_id: (Turkish, Spanish)
TRANSLATIONS = {
    "MAMS-EN-000165-FOOD": (
        "Yemek boyunca aklımdan yalnızca ‘ortalama, ortalama, ortalama’ geçiyordu; hesap gelince onun da ortalamanın biraz üzerinde olduğunu gördüm.",
        "Durante toda la comida no dejé de pensar ‘normal, normal, normal’; cuando llegó la cuenta, también estaba un poco por encima de la media.",
    ),
    "MAMS-EN-000793-FOOD": (
        "Kötü servis, ortalama yemekler ve sıradan menü seçenekleri olmasa fiyat o kadar da kötü sayılmazdı.",
        "El precio no estaría tan mal si no fuera por el mal servicio, la comida normal y las opciones de menú poco inspiradas; de postre había tiramisú o fresas con nata.",
    ),
    "MAMS-EN-001004-FOOD": (
        "Yemekler idare eder ama pek akılda kalıcı değil; ortam özellikle bir buluşma için çok gürültülüydü ve ikinci içecek siparişimizi yirmi dakika sonra hatırlattığımızda garson bizi hiç görmemiş gibiydi.",
        "La comida está bien, aunque no destaca; el ambiente era muy ruidoso, sobre todo para una cita, y cuando recordamos al camarero veinte minutos después que habíamos pedido otra ronda, parecía no habernos visto antes.",
    ),
    "MAMS-EN-001428-FOOD": (
        "Ortalama yemekler, kabarık bir hesap ve servisten pek hoş olmayan tavırlar bekleyin.",
        "Espera comida normal, una cuenta elevada y un trato poco agradable por parte del servicio.",
    ),
    "MAMS-EN-001950-FOOD": (
        "Porsiyonlar küçük, yemekler ortalama. Hesabı istedik; garson ‘tamam’ dedi, beş dakika sonra da kahve ya da tatlı isteyip istemediğimizi sordu.",
        "Las porciones son pequeñas y la comida es normal. Pedimos la cuenta; el camarero dijo que sí y, cinco minutos después, preguntó si queríamos café o postre.",
    ),
    "MAMS-EN-001951-FOOD": (
        "Yemekler ortalama, fiyatlar yüksek ve genel deneyim oldukça moral bozucuydu.",
        "La comida es normal, los precios son altos y la experiencia en conjunto fue bastante decepcionante.",
    ),
    "MAMS-EN-002173-FOOD": (
        "Yemekler kesinlikle idare ederdi; ancak fiyatlar her ölçüte göre yüksekti ve dört kişilik, içeceksiz akşam yemeği 300 doların üzerindeydi.",
        "La comida estaba bien, aunque los precios eran altos se mirara por donde se mirara: una cena para cuatro, sin bebidas, superó los 300 dólares.",
    ),
    "MAMS-EN-002395-FOOD": (
        "Sevimli bir restoran; yemekler ortalama ve menü küçük.",
        "Es un restaurante bonito, con comida normal y una carta pequeña.",
    ),
    "MAMS-EN-002497-FOOD": (
        "Yönetim her zaman kabaydı ama iyi personel bunu dengeliyordu; o personelin desteği olmayınca Jesse’s, zayıf dekoru ve ortalama yemekleriyle sıradan bir pubdan ibaret.",
        "La gerencia siempre fue desagradable, pero el buen personal lo compensaba; sin ese equipo, Jesse’s no es más que otro pub con decoración pobre y comida normal.",
    ),
    "MAMS-EN-002982-FOOD": (
        "Yemekler ortalama; servis ise ortalamanın çok altında. Garson özel yemekleri anlatırken sürekli takılıyor ve gecenin büyük kısmında bardaklarımız boş kalıyordu; ortam da müşteri kitlesi de oldukça yapaydı.",
        "La comida es normal, pero el servicio está muy por debajo de la media. El camarero se trababa al explicar los especiales y dejó nuestras copas vacías durante gran parte de la noche; el ambiente y el público también resultaban artificiales.",
    ),
    "HA-CAND-EN-ATMOSPHERE-0001": ("Yemek salonunda öğle saatlerinde tavandan aydınlatma kullanılıyor.", "Durante el almuerzo, el comedor utiliza luces de techo."),
    "HA-CAND-EN-ATMOSPHERE-0004": ("Akşam yemeği boyunca arka planda enstrümantal müzik çalıyor.", "Durante la cena suena música instrumental de fondo."),
    "HA-CAND-EN-ATMOSPHERE-0007": ("Masalar doldukça konuşma uğultusu artıyor.", "El ruido de las conversaciones aumenta a medida que se ocupan más mesas."),
    "HA-CAND-EN-ATMOSPHERE-0010": ("Bir duvar boyunca localar, ortada ise iki kişilik masalar bulunuyor.", "Hay bancos corridos junto a una pared y mesas para dos en el centro."),
    "HA-CAND-EN-ATMOSPHERE-0013": ("Hava uygun olduğunda teras servise açılıyor; ana yemek salonu içeride kalıyor.", "La terraza abre cuando el tiempo lo permite, mientras que el comedor principal permanece en el interior."),
    "HA-CAND-EN-ATMOSPHERE-0016": ("Saat altıda masaların yaklaşık yarısı doluydu.", "A las seis, aproximadamente la mitad de las mesas estaban ocupadas."),
    "HA-CAND-EN-ATMOSPHERE-0019": ("Ana salonda on iki masa ve altı kişilik bir tezgâh var.", "La sala principal tiene doce mesas y una barra con seis asientos."),
    "HA-CAND-EN-ATMOSPHERE-0022": ("Öğleden sonra yemek salonunda tavan vantilatörleri çalışıyor.", "Por la tarde funcionan los ventiladores de techo del comedor."),
    "HA-CAND-EN-ATMOSPHERE-0029": ("Oturduğumuz andan tatlıya kadar ışıkların oldukça loş tutulduğunu fark ettim.", "Noté que las luces se mantuvieron bastante tenues desde que nos sentamos hasta el postre."),
    "HA-CAND-EN-ATMOSPHERE-0030": ("Akşam yemeği kalabalığı geldikçe salonun daha hareketli hâle geldiğini seslerden anlayabiliyorduk.", "Se notaba por el ruido que la sala estaba cada vez más concurrida a medida que llegaba la gente para cenar."),
    "HA-CAND-EN-ATMOSPHERE-0031": ("Yemek boyunca müziğin seviyesi hemen hemen aynı kaldı.", "El volumen de la música se mantuvo más o menos igual durante toda la comida."),
    "HA-CAND-EN-ATMOSPHERE-0032": ("Saat yediye gelindiğinde masaların çoğu doluydu; barın yanında birkaç boş yer kalmıştı.", "A las siete, la mayoría de las mesas estaban ocupadas y quedaban algunos asientos libres junto a la barra."),
    "HA-CAND-EN-ATMOSPHERE-0033": ("Yemek alanında duvar boyunca localar ve iki kişilik küçük masalar var.", "En la zona de comedor hay bancos corridos junto a una pared y pequeñas mesas para dos."),
    "HA-CAND-EN-ATMOSPHERE-0034": ("İçeceklerimiz bardan geldi; yemek salonu açık mutfağın çevresine uzanıyordu.", "Las bebidas vinieron de la barra y el comedor se extendía alrededor de una cocina abierta."),
    "MAMS-EN-000544-AMBIENCE": ("Müzik aynı anda hem house tarzı radyo yayınından hem de canlı gruptan geliyor.", "Sonaban al mismo tiempo música house de la radio y una banda en directo."),
    "MAMS-EN-000556-AMBIENCE": ("Nostaljik dağ evi tarzı süsler hareketli ön salonu ve barı kaplıyor; loş, sigara içilmeyen arka salon ise vitraylar ve modern Mission tarzı sandalyelerle döşenmiş.", "Una decoración nostálgica de estilo refugio de montaña adorna el comedor delantero y la barra, más concurridos; la sala trasera, con luz tenue y para no fumadores, tiene vidrieras y sillas modernas de estilo Mission."),
    "MAMS-EN-001960-AMBIENCE": ("Büyük kalabalıkları alabilecek uzun bir bar alanı ve maç izlemek için duvarlarda büyük ekranlar var; ekranlar ortama baskın gelmiyor. Ön tarafta birkaç loca ve çalışan bir şömine bulunan geniş bir oturma alanı bulunuyor.", "Hay una barra larga con espacio suficiente para mucha gente y pantallas grandes en las paredes para ver partidos, aunque no dominan el ambiente. En la parte delantera hay una amplia zona de estar con varios bancos y una chimenea en funcionamiento."),
    "MAMS-EN-002555-AMBIENCE": ("Loş ışıkta porsiyonların etkileyici göründüğünü düşündüm ve yemeğe başladım.", "Con la luz tenue, las porciones me parecieron impresionantes y empecé a comer."),
    "HA-CAND-EN-SERVICE-0001": ("Garson, yemek siparişini kaydetmeden önce masanın içecek siparişini alıyor.", "El camarero toma primero el pedido de bebidas de la mesa y después anota el pedido de comida."),
    "HA-CAND-EN-SERVICE-0005": ("Listeye adımızı yazdırmamızla masaya oturmamız arasında yaklaşık on dakika bekledik.", "Esperamos unos diez minutos desde que nos apuntamos en la lista hasta que nos sentaron."),
    "HA-CAND-EN-SERVICE-0009": ("Karşılama görevlisi kişi sayısını doğruladıktan sonra bir masa belirliyor.", "El personal de recepción asigna una mesa después de confirmar el número de personas."),
    "HA-CAND-EN-SERVICE-0013": ("Rezervasyon girişte isimle doğrulandı.", "La reserva se confirmó en la entrada usando el nombre."),
    "HA-CAND-EN-SERVICE-0017": ("Başlangıçlar ana yemeklerden önce masaya getiriliyor.", "Los entrantes llegan a la mesa antes que los platos principales."),
    "HA-CAND-EN-SERVICE-0021": ("Diğer içecekler getirilmeden önce masaya su konuyor.", "Ponen agua en la mesa antes de traer las demás bebidas."),
    "HA-CAND-EN-SERVICE-0024": ("İstendiğinde hesap masaya getiriliyor.", "La cuenta se lleva a la mesa cuando la solicitan."),
    "HA-CAND-EN-SERVICE-0028": ("Müşteriler tepsileri ve çatal bıçakları self-servis noktasından alıyor.", "Los clientes recogen las bandejas y los cubiertos en la zona de autoservicio."),
    "HA-CAND-EN-SERVICE-0043": ("Menüye birkaç dakika baktıktan sonra içecek ve yemek siparişlerimizi birlikte verdik.", "Miramos la carta unos minutos y después pedimos las bebidas y la comida a la vez."),
    "HA-CAND-EN-SERVICE-0045": ("Girişte kayıt yaptırdık; kişi sayısı doğrulandıktan sonra masamıza götürüldük.", "Nos presentamos en la entrada y, tras confirmar el número de personas, nos acompañaron a la mesa."),
    "HA-CAND-EN-SERVICE-0046": ("Karşılama bankosunda adımı söyledim; bizi masaya götürmeden önce ismimi rezervasyondaki kayıtla eşleştirdiler.", "Di mi nombre en el mostrador de recepción y lo comprobaron con la reserva antes de sentarnos."),
    "HA-CAND-EN-SERVICE-0049": ("Önce başlangıçlarımız geldi; tabaklar kaldırıldıktan sonra ana yemekler getirildi.", "Primero llegaron los entrantes y, después de retirar los platos, trajeron los platos principales."),
    "HA-CAND-EN-SERVICE-0052": ("Hesabı istedik ve masada kart terminaliyle ödeme yaptık.", "Pedimos la cuenta y pagamos en la mesa con el datáfono."),
    "HA-CAND-EN-SERVICE-0054": ("Çatal bıçağımızı kendimiz aldık ve tepsileri toplama alanına bıraktık.", "Recogimos nuestros cubiertos y devolvimos las bandejas a la zona de recogida."),
    "HA-CAND-EN-SERVICE-0055": ("Tezgahta sipariş verdim, masa numarası aldım ve numaranın çağrılmasını bekledim.", "Pedí en el mostrador, cogí un número de mesa y esperé a que lo llamaran."),
    "HA-CAND-EN-SERVICE-0059": ("Yan tezgahta teslim alma numaramı söyledim ve numara çağrılınca paketimi teslim aldım.", "Di el número de recogida en el mostrador lateral y recogí la bolsa cuando llamaron mi número."),
    "HA-CAND-EN-SERVICE-0060": ("Girişteki listeye adımızı yazdık ve görevlinin çağırmasını bekledik.", "Apuntamos nuestro nombre en la lista de la entrada y esperamos a que el personal de recepción nos llamara."),
    "HA-CAND-EN-SERVICE-0061": ("Ana yemekler getirilmeden önce başlangıç tabaklarımız kaldırıldı.", "Retiraron los platos de los entrantes antes de traer los platos principales."),
    "HA-CAND-EN-SERVICE-0062": ("Listeye katılmamızdan masaya geçmemize kadar süreç yaklaşık on beş dakika sürdü.", "Desde que nos apuntamos hasta que llegamos a la mesa pasaron unos quince minutos."),
    "HA-CAND-EN-SERVICE-0063": ("Garson önce içecek siparişini aldı, sonra yemek siparişi için döndü ve istediğimizde hesabı getirdi.", "El camarero tomó primero el pedido de bebidas, volvió para anotar el pedido de comida y trajo la cuenta cuando se la pedimos."),
    "HA-CAND-EN-SERVICE-0064": ("Taşınabilir cihaza kartımı okuttum; fiş masada basıldı.", "Acerqué la tarjeta al datáfono y el recibo se imprimió en la mesa."),
    "HA-CAND-EN-SERVICE-0065": ("Tezgahta sipariş verdik, basılı numarayı sakladık ve numaramız ekranda görününce tepsimizi aldık.", "Pedimos en el mostrador, guardamos el número impreso y recogimos la bandeja cuando apareció en la pantalla."),
    "HA-CAND-EN-VALUE-0001": ("Menü fiyatları, aynı mutfağı sunan yakındaki restoranların fiyatlarına benzer.", "Los precios de la carta son similares a los de otros restaurantes cercanos de la misma cocina."),
    "HA-CAND-EN-VALUE-0005": ("Ana yemek fiyatları, bu bölgedeki restoranların ortalamasına yakın.", "Los precios de los platos principales están cerca de la media de esta zona."),
    "HA-CAND-EN-VALUE-0009": ("Listelenen fiyatlar, orta sınıf bir restoran için beklenen aralıkla örtüşüyor.", "Los precios indicados están dentro del rango esperado para un restaurante de precio medio."),
    "HA-CAND-EN-VALUE-0017": ("Öğle setinin fiyatı, yakındaki diğer iki tabaklı menülerle karşılaştırılabilir.", "El precio del menú de almuerzo es comparable al de otros menús de dos platos de restaurantes cercanos."),
    "HA-CAND-EN-VALUE-0020": ("Bu noodle yemeğinin fiyatı, bu tür yemekler için tipik olan aralıkta.", "El precio de este plato de fideos está dentro del rango habitual para este tipo de plato."),
    "HA-CAND-EN-VALUE-0039": ("Set menü, üç çeşit akşam yemeği için beklediğim fiyat aralığındaydı.", "El menú cerrado estaba en el rango de precio que esperaba para una cena de tres platos."),
    "HA-CAND-EN-VALUE-0041": ("Bu tür bir noodle yemeği için genelde ödediğim seviyede bir fiyat ödedim.", "Pagué más o menos lo que suelo pagar por un plato de fideos de este tipo."),
    "HA-CAND-EN-VALUE-0043": ("Yakındaki birkaç menüyü karşılaştırdık; öğle setleri benzer bir fiyat aralığındaydı.", "Comparamos las cartas de varios restaurantes cercanos y los menús de almuerzo estaban en un rango de precios parecido."),
    "HA-CAND-EN-VALUE-0050": ("Fiyatlar, yakında gittiğimiz diğer İtalyan restoranlarıyla yaklaşık aynıydı.", "Los precios eran aproximadamente los mismos que en los otros restaurantes italianos cercanos que visitamos."),
    "HA-CAND-EN-VALUE-0051": ("Ana yemek fiyatları, şehrin o bölgesindeki ortalama fiyatlar civarındaydı.", "Los precios de los platos principales estaban alrededor de la media local de esa zona."),
    "HA-CAND-EN-VALUE-0052": ("Bu tür bir mahalle restoranındaki bir öğün için genelde ödediğim kadar ödedim.", "Pagué aproximadamente lo que suelo pagar por una comida en un restaurante de barrio de este tipo."),
    "HA-CAND-EN-VALUE-0053": ("Menü, bu tarz rahat deniz ürünleri restoranlarında gördüğüm olağan fiyat aralığındaydı.", "La carta estaba en el rango de precios habitual que suelo ver en restaurantes informales de marisco."),
    "HA-CAND-EN-VALUE-0054": ("Fiyatlar, yakınlarda baktığımız benzer set menülerle karşılaştırılabilir düzeydeydi.", "Los precios eran comparables a los de otros menús similares que vimos cerca."),
    "HA-CAND-EN-VALUE-0055": ("Yemek hesabımız, bölgede denediğimiz diğer yerlerle yaklaşık aynıydı.", "Nuestra comida costó aproximadamente lo mismo que en otros sitios que probamos en la zona."),
    "HA-CAND-EN-VALUE-0056": ("Akşam yemeği, orta segment ve üç çeşit sunan bir restorandan beklediğim fiyat aralığıyla örtüştü.", "El precio de la cena estuvo dentro del rango que esperaba para una cena de tres platos en un restaurante de precio medio."),
    "HA-CAND-EN-VALUE-0057": ("Kahve ve tatlının fiyatları, aynı meydandaki kafelerdeki fiyatlara benzerdi.", "El café y el postre tenían precios similares a los de las cafeterías de la misma plaza."),
    "HA-CAND-EN-VALUE-0058": ("Yakındaki restoranların dışarıda sergilediği menülere göre öğle yemeği fiyatları bölge ortalaması civarındaydı.", "Según las cartas expuestas en los restaurantes cercanos, los precios del menú de mediodía estaban alrededor de la media de la zona."),
    "HA-CAND-EN-VALUE-0059": ("Bu noodle yemeğinin fiyatı, başka yerlerde aynı tür yemek için normalde ödediğim kadardı.", "Este plato de fideos costó más o menos lo que suelo pagar por el mismo tipo de plato en otros sitios."),
    "HA-CAND-EN-VALUE-0060": ("Alakart fiyatlar, mahalledeki benzer restoranların menü fiyatlarıyla aynı seviyedeydi.", "Los precios a la carta estaban en línea con los de restaurantes similares del barrio."),
}


def read_verified_sources() -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    with MAMS_PATH.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["human_decision"] == "ACCEPT":
                sources[row["candidate_id"]] = {
                    "english_sentence": row["sentence"],
                    "mapped_aspect": row["human_final_aspect"],
                    "source_group_id": row["source_group_id"],
                    "original_source_type": "mams_gold_annotation",
                    "original_source_dataset": "justinsiow/mams-acsa",
                    "original_source_license": "apache-2.0",
                }
    with HUMAN_PATH.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["human_decision"] == "ACCEPT" and row["human_verified"] == "true":
                sources[row["candidate_id"]] = {
                    "english_sentence": row["sentence"],
                    "mapped_aspect": row["human_final_aspect"],
                    "source_group_id": row["source_group_id"],
                    "original_source_type": row["source_type"],
                    "original_source_dataset": "human_authored_neutral_candidates_en",
                    "original_source_license": "not_applicable_project_authored",
                }
    return sources


def identities(source_id: str, aspect: str, language: str) -> tuple[str, str]:
    if source_id.startswith("HA-CAND-EN-"):
        suffix = source_id.removeprefix("HA-CAND-EN-")
        return f"HA-CAND-{language.upper()}-{suffix}", f"HA-{suffix}"
    row_number = source_id.split("-")[2]
    token = aspect.upper()
    return (
        f"MAMS-CAND-{language.upper()}-{token}-{row_number}",
        f"MAMS-{token}-{row_number}",
    )


def build_rows(language: str, sources: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for aspect in ("taste", "atmosphere", "service", "value"):
        for source_id in SELECTED_IDS[aspect]:
            source = sources[source_id]
            if source["mapped_aspect"] != aspect:
                raise ValueError(f"Aspect mismatch for {source_id}")
            candidate_id, parallel_id = identities(source_id, aspect, language)
            translated = TRANSLATIONS[source_id][0 if language == "tr" else 1]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "parallel_source_group_id": parallel_id,
                    "english_source_id": source_id,
                    "english_sentence": source["english_sentence"],
                    "translated_sentence": translated,
                    "language": language,
                    "mapped_aspect": aspect,
                    "proposed_sentiment": "neutral",
                    "original_source_type": source["original_source_type"],
                    "original_source_dataset": source["original_source_dataset"],
                    "original_source_license": source["original_source_license"],
                    "translation_human_verified": "false",
                    "human_decision": "",
                    "human_final_aspect": "",
                    "human_final_sentiment": "",
                    "review_notes": "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            old_rows = list(csv.DictReader(file))
        if any(row.get("human_decision", "") for row in old_rows):
            raise FileExistsError(f"Refusing to overwrite reviewed translations: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def review_batch(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_source = {row["english_source_id"]: row for row in rows}
    batch = [by_source[source_id] for source_id in REVIEW_SOURCE_IDS]
    random.Random(42).shuffle(batch)
    return batch


def verified_english_coverage() -> dict[str, int]:
    original = Counter()
    with (PROJECT_ROOT / "data" / "cleaned" / "train.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            if row["language"] == "en" and row["sentiment"] == "neutral":
                original[row["aspect"]] += 1
    sources = read_verified_sources()
    expansion = Counter(source["mapped_aspect"] for source in sources.values())
    return {
        aspect: original[aspect] + expansion[aspect]
        for aspect in ("taste", "atmosphere", "service", "value")
    }


def main() -> None:
    selected_flat = {source_id for values in SELECTED_IDS.values() for source_id in values}
    if set(TRANSLATIONS) != selected_flat:
        missing = selected_flat - set(TRANSLATIONS)
        extra = set(TRANSLATIONS) - selected_flat
        raise ValueError(f"Translation mapping mismatch: missing={missing}, extra={extra}")

    sources = read_verified_sources()
    unavailable = selected_flat - set(sources)
    if unavailable:
        raise ValueError(f"Selected sources are not verified: {sorted(unavailable)}")

    tr_rows = build_rows("tr", sources)
    es_rows = build_rows("es", sources)
    expected = Counter({"taste": 10, "atmosphere": 18, "service": 22, "value": 19})
    if Counter(row["mapped_aspect"] for row in tr_rows) != expected:
        raise ValueError("TR source quotas are invalid")
    if Counter(row["mapped_aspect"] for row in es_rows) != expected:
        raise ValueError("ES source quotas are invalid")
    if {
        (row["parallel_source_group_id"], row["english_source_id"])
        for row in tr_rows
    } != {
        (row["parallel_source_group_id"], row["english_source_id"])
        for row in es_rows
    }:
        raise ValueError("TR/ES parallel identities do not match")

    write_csv(TR_PATH, tr_rows)
    write_csv(ES_PATH, es_rows)
    tr_batch = review_batch(tr_rows)
    es_batch = review_batch(es_rows)
    write_csv(TR_BATCH_PATH, tr_batch)
    write_csv(ES_BATCH_PATH, es_batch)

    coverage = verified_english_coverage()
    status = (
        "COMPLETE" if all(count >= 25 for count in coverage.values()) else "INCOMPLETE"
    )
    summary = [
        f"EN_NEUTRAL_EXPANSION_V1 = {status}",
        "Verified EN:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "TR candidates prepared:",
        *[f"{aspect}: {expected[aspect]}" for aspect in expected],
        "ES candidates prepared:",
        *[f"{aspect}: {expected[aspect]}" for aspect in expected],
        f"TR Batch 1: {len(tr_batch)}",
        f"ES Batch 1: {len(es_batch)}",
        "translation human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
