# M-ABSA Label Yapısı

M-ABSA'nın her ham satırı şu biçimdedir:

```text
cümle####[[aspect term, aspect category, sentiment polarity], ...]
```

- **Aspect term:** Cümlede değerlendirilen açık ifade; örneğin `coffee`.
- **Aspect category:** İfadenin ait olduğu genel konu; örneğin `food quality`.
- **Sentiment polarity:** O aspect hakkındaki duygu yönü; örneğin `positive`, `negative` veya `neutral`.

Bir cümlede birden fazla triplet bulunabilir. `NULL`, cümlede açıkça yazılmayan ancak yorumdan anlaşılan bir aspect termi gösterebilir.

Bu aşamada etiketler olduğu gibi incelenir. Dönüştürme veya label mapping yapılmaz.
