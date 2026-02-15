
# Comment run?

Par exemple : 
```sh
python -m src.experiment.complexity
```


# Implémentations

Le répertoire module contient 3 tentatives d’implémentation du module MoM. 2 versions expérimentales varlen qui suivent l'implémentation hardware-efficient du papier, utilisées pour évaluer la vitesse d’exécution. Une version naïve implémenté 100% en torch utilisé pour les entraînements et les experiences (car ne nécéssitant pas d'écrire le backward).

- module.naive_mom : Dans l'implémentation naive on parcourt la séquence, et pour chaque batch de token on calcule la passe-avant complète des mémoire. Ensuite en utilisant des gather et des opérations scatter on utilise seulement celles qui ont été routé. Cette implémentation est plus lente mais gère le backward avec graph de calcul de torch.

- module.old.mom_varlen : C’est la toute première implémentation en varlen. Elle fonctionne, mais n’est pas du tout optimisée en mémoire. Le principal problème vient de la matérialisation de x_tilde et o_tilde, ce qui duplique x autant de fois qu’il y a de mémoires, entraînant une forte consommation mémoire. On y a implémenté les kernels forward pour la linear attention et la GLA, ainsi que le backward de la linear attention. Cette version a servi de base au début du projet avant de passer à une solution plus efficace.

- module.mom_fast : Il s’agit de la version la plus optimisée à ce jour. Elle reprend les grandes lignes de old.mom_varlen, mais sans matérialiser x_tilde et o_tilde. À la place, on utilise m_ids, t_orig et b_orig pour accéder directement aux bonnes positions dans x (qui garde sa forme (T, B, D)) et on écrit directement dans o. Cette écriture directe rend l’exécution un peu plus lente à cause des accès concurrents (plusieurs kernels écrivant potentiellement sur la même case de o), gérés via tl.atomic_add. Malgré cela, cette version est bien plus économe en mémoire.

Il reste un problème pour ces deux implémentation c'est qu'on calcule à l'avance q, k et v pour tous les tokens ce qui reviens à faire comme les transformers, ce qui coûte cher en mémoire. Dans le papier ils calculent les k, q et v à la volé ce que nous n'avons pas fait car cela aurait demandé de faire un produit matriciel en triton et nous n'avions malheureusement plus le temps.

## Ce qu'on a fait

- On a réimplémenté MoM dans 3 versions différentes dont plusieurs fonctionnant avec un kernel triton. 

- Nous avons implémenté les updates en naive pour la **linear attention**, le **retnet**, le **GLA** et **G-deltanet**. 

- Nous avons implémenté les updates en triton pour la **linear attention** (backward et forward), le **GLA** (forward uniquement)

Pour les expériences, nous avons utilisé des librairies tel que fla ou encore transformers avec les implémentations llama pour les baselines.

# Erreur critique MQAR

Suite à la présentation, les examinateurs ont été surpris par les résultats obtenus sur MQAR. Après vérification, nous avons identifié une erreur critique dans le code de l’expérience : les labels envoyés par le dataloader étaient incorrects.
Par conséquent, les résultats présentés sur le poster ne sont pas valides et ne doivent pas être pris en compte. L’erreur a été corrigée, mais les expériences n’ont pas encore été relancées.
