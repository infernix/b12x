# Lossless BF16 PCIe two-shot artifact map

Status: **qualified test artifacts** for the four-rank CUDA-graph and eager
correctness run documented in
[`pcie_twoshot_bf16_sm120.md`](pcie_twoshot_bf16_sm120.md).

The isolated compile cache contained 36 CUTLASS DSL objects: three collective
operations, four ranks, and three slot-selection modes per rank. `eager` means
host-selected double buffering; `graph-0` and `graph-1` mean device-selected
double buffering with the stated synchronized slot bias. Every manifest binds
Python 3.12.3, PyTorch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2,
cuda-bindings 13.3.1, PTXAS 13.3.73, 512 threads, and 4,096 BF16 elements per
row. The package fingerprint recorded by every manifest is
`4f450276a5f1e75f7037b7510d6d5e2ac25b81d4a7ec53c45214775cc3aa8b4b`.

| Operation | Rank | Physical GPU UUID | Slot mode | Manifest SHA-256 | Object SHA-256 |
|:--|--:|:--|:--|:--|:--|
| all-gather | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | eager | `9275b890ab13909d205c7505496ab1c6d8cc97110d268b5dbe9981813bdd31d0` | `8009d90a910891a15db3175daf58dce39fbe79f7eaa8b1613861be2fa2200fc5` |
| all-gather | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-0 | `82306d1092e5213f853388dabe65faab1339f86d36b570fa5acd68464ce86836` | `bf19906d5d571d0d4c427c2affed4105311f489ab56579da7d11577d0a606ddc` |
| all-gather | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-1 | `81d1f694e4d128d25fbe5b694e535366b7542f1e28718ea4d203fa547926739a` | `2615e3d04fd09260adf6ca40fdad501c83390eb5be99b130880358b6f70a4779` |
| all-gather | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | eager | `28822e433e8338c0034dc07328c75438b20fdd90ec8c67909c6fbaeb78d96dbd` | `b2fd3d6beffe4ae4f41a9a291f7cfda21cf0815619c989091dc952393e318319` |
| all-gather | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-0 | `e5820d5ba37534fe7e8b4b11981b9532709fe1f85b7ac9dbec757be033df2abc` | `a69cf052cffe69c7bea0a3e9f69eab7ac64687b88ca45bc65dc2619b3f761191` |
| all-gather | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-1 | `77c9acfe81838f09e5b17a1f8c195ea08de08ba971dc14e8f5431224736493ab` | `e902a07dd52457cf8ac74b088714f95f9414a6aa25224f327fc9fdaac8f1276e` |
| all-gather | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | eager | `eee5c1f51e9319d5129efa24c7cb3e6bb45c7d5b0bef9348a9eaf4805df86610` | `a08b529ee887b0db6d2ba808a6944e488730f15f3af43a1642a8e7be526be786` |
| all-gather | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-0 | `4570d73fe0be0f4e82ad50abe67bbf7c4995f9e798bef34962ad089a6f27cb53` | `eb3eb25c570c3005accf0f32f5df5cd65a864de0a5f0f4da81bffb9bb46cda7a` |
| all-gather | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-1 | `e4492df2846fbf275c5d905eb25b8c4266127ad5b59c8b39264d4efb0acd2364` | `97327954f16053c289ffd9d58494cfa273dc2d3ea939884a9127b96297ad8fca` |
| all-gather | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | eager | `b55379c15b6a7e6729e2438c3ab0e142852ed34abcd82ea6499abe7953faa880` | `e19fdf9dc3883d5ce8fb3e265201d71820ee6e9e78ac7a724fd45c4ef80ac073` |
| all-gather | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-0 | `a17f1e4fa3d6389093db395eab232b1acb11f97e1c029cd6bdd6ea3d557a370d` | `8749d80caab6f6994dfa8e72c8a7b28bfe778cc157f30423c653cd36645e00e6` |
| all-gather | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-1 | `59bf1e1ded2bd82cb07392140d0dc91d813aea4ccbc6fdedb7a6b1e5c8ea4e8b` | `178242ecc2c5671e48a30c9d289b6374c77eafbf3fc7b6a8f3187e0010cbb713` |
| pull all-reduce | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | eager | `cccc26cd43334dc39868d43ef064d288590f64860b728c666d0a6bd471ba7c72` | `e1f8b48f375f40280ebf42d6d41e77539e97118224327aee55f881aa8c27c832` |
| pull all-reduce | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-0 | `021ed1f4354ca6d74cea5da6e04d4194dccfc5978ca0aca75575997e5461cbdb` | `b2ffd95f7445b3a703b069218824b3b51fe6fffd63bd49dc6b0cb2a04a682781` |
| pull all-reduce | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-1 | `05710d590f87652ad74de4b87833c13db6c7d34c85dcc89cfb3658743ded7cb3` | `81b2e72959ad2b8b651780e4b65d632d7dc1c20a5173d59a196570d9b2b00f94` |
| pull all-reduce | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | eager | `32d4efa7304335d0d35fd2adae3c8aed89b355d600f3a8a8f617c55113c75ad2` | `204b34eab7f232b8cbfa2ada4674260c09ddaa77fac3b1c4f54a815a4fab95a7` |
| pull all-reduce | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-0 | `96de75cd7ec94568f4902333138f02a8935380abd049693db323015cc32651bc` | `e92a41823c79b80c3a6fcccc1926fa2f57c3a1981f3ff91b675a6c57481f91d0` |
| pull all-reduce | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-1 | `ba60d0d553a3269f1cbedbebde5f189f7c4985fda9c2efc2d35f414229cec752` | `bd07f3b69fc486714463c96be367fa4f8a3c298c138c9ec9fa453da677f3c399` |
| pull all-reduce | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | eager | `9f24ff2b5f3e4c9cc2e876bd1bd9734130df2ee97c8a99f3f2534fce0c87e535` | `176400bacc6f849b75c54778dd841b54a3e8a6d7eb2355d651a62207d21eba69` |
| pull all-reduce | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-0 | `bb08c4000b543a1499781d517e4b03bc74e44fdf1ba07cefc7b648555696ea66` | `5685786d1c072968abf5f61afe8ee01ce7da8b6c137d4f27960e752c43d5530d` |
| pull all-reduce | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-1 | `2b04473b5acc0eb1e085a5c1988e39912f93bc3a834a51cadd183f0a583eed65` | `907f1e4c1e254b0d36f6001470d5c4f69c65b589883e5d79e5f815458f80b7bc` |
| pull all-reduce | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | eager | `908ae800365a01ef12de27ade93598fd64237f91f13aeda9a9f8304def60c5ab` | `3c752a70967f03aade1dbf084ed1dc22f9a4ea908a09c33b528ffb84b7e3e120` |
| pull all-reduce | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-0 | `a2518e1b4974a5da73b13656c7f9aa5209c039ef298e61d1ca29f2f9c6b54d75` | `8f5a3209ee9c3ad23411c02160d20e4893266ff624eb8b90748842b17ab1026a` |
| pull all-reduce | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-1 | `343230ed4b11e6c9a7d6d466a8117184d8a539360952c1f8a102869f8bf53825` | `bae126504838356b5eb1a12c69cbecd3fb91ced4600ce84ba6abfb36fb870f40` |
| reduce-scatter | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | eager | `8ae438076ba48942606dd356ff469824267abf391d9c58622b081afdd34f1036` | `cbb57bec65df5f6be05595158e53e2d990d84f93afd37c1eda93cdf8229c122a` |
| reduce-scatter | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-0 | `7ad8538a85f9a5f5bc30a3f4969d8cde09a13f86659024389c19b3d14b388bc1` | `713bd307a851f5f5ed18658b0b2b1dccb385edfd4d1da4718a5db031636461fe` |
| reduce-scatter | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-1 | `1a0a51bd14fed37960798d6e18c9bba95fe47aef6d1b7ff97d83054a50320617` | `7e5c33c9329d965702ef481e2b060ac64d8b9021b5e6835aef121a5a9042aac0` |
| reduce-scatter | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | eager | `aa1b1b150f8d30a6c5eeaccc1040084b32910d83c7c0317d1ef2da33853528f6` | `ba388a6fdc194d047d95c3c3a6914a0832c57045f9d73cacde0fa46a8cafe999` |
| reduce-scatter | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-0 | `74f5ab607b27bc25448f6b4b7980fa8ba7fb04e82a7b2f3bfa385ab3882c1adb` | `f1f04d6b39319be060fd992492e486a769ca5ecee05bb8fa6cfe9870e75b0d9a` |
| reduce-scatter | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-1 | `6e314931be63898c7cb7a5f5b52e3a36f6a2176f53ce74d9237822bfeffb9cd7` | `78eab4c758e67bddc100ce648d466ffea7f76bfffcd9460c045f0e2ce62be577` |
| reduce-scatter | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | eager | `2683cf4e1a0fa9605a9bef0c3ca49251f6db76982dee1fd2b9633a29109ce60c` | `926c761cfdaddbbe8a592893cb5a9687005fc94b26bbbf54990a453aeb201f99` |
| reduce-scatter | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-0 | `e6ccb74299ad9108c98a820c4cd86c66fb8e46e1f0f00c00fe40c8ae0616e6e2` | `9fd9be653f26b533d8fd2f9d4538dc3c0a823a9bb3827399d8a9910832bb1eb2` |
| reduce-scatter | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-1 | `2606d2966d0f110ab23c150f9c98371be64c85f92b794eab8a2ea60ab45e6ea9` | `c9efc7267f1cfd3b3f419fe9b3d1c3d1d90e502f1743dc6054d2bf1d841f516c` |
| reduce-scatter | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | eager | `73eb270164c203b4613b6091a77b05a39d721d0c626dac3f1fa2e50e7e407fd7` | `e09f0ac20b1dac6796c8b8223cffc5c3088116a614eacb01e6df17cbca0969da` |
| reduce-scatter | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-0 | `6ef12f4c064a603b2cdd1c18dcd5ac21c6b72445034a1cffda31758f970bcd5f` | `84c49e18e4845e6bdd42b656c2cc55b041c20f52979b8066f60a48224a703245` |
| reduce-scatter | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-1 | `d35c0c771b55bf10d036fe5c30a97430a684d15a39663e3839691509a6375291` | `4447c464f163b4f673785b343ffb1df9ebd5402233366236c865fc1943075ac2` |
