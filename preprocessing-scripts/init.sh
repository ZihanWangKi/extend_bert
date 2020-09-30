# clone the BERT repo to current directory
git clone https://github.com/google-research/bert.git

# download the multilingual vocabulary
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt

# create a dummy text file & folder for testing
mkdir -p test_data_folder
mkdir -p test_data_folder/raw_txt
mkdir -p test_data_folder/txt

for _ in $(seq 0 1 9); do
  echo "Anarchism
  Anarchism is an anti-authoritarian political philosophy that advocates self-managed, self-governed societies based on voluntary, cooperative institutions and the rejection of hierarchies those societies view as unjust.
  These institutions are often described as stateless societies, although several authors have defined them more specifically as distinct institutions based on non-hierarchical or free associations.
  Anarchism's central disagreement with other ideologies is that it holds the state to be undesirable, unnecessary, and harmful.
  Anarchism is usually placed on the far-left of the political spectrum, and much of its economics and legal philosophy reflect anti-authoritarian interpretations of communism, collectivism, syndicalism, mutualism, or participatory economics.
  As anarchism does not offer a fixed body of doctrine from a single particular worldview, many anarchist types and traditions exist and varieties of anarchy diverge widely.
  Anarchist schools of thought can differ fundamentally, supporting anything from extreme individualism to complete collectivism.
  Strains of anarchism have often been divided into the categories of social and individualist anarchism, or similar dual classifications.
  The etymological origin of anarchism derives from ancient Greek word "anarkhia".
  "Anarkhia" meant "without a ruler" as it was composed by the prefix a (i.e.
  "without") and the word "arkhos" (i.e.
  leader or ruler).
  The suffix -ism is used to denote the ideological current that favours anarchism.
  The first known   use of this word was in 1642.
  Various factions within the French Revolution labelled opponents as anarchists although few shared many views of later anarchists.
  There would be many revolutionaries of the early 19th century who contributed to the anarchist doctrines of the next generation, such as William Godwin and Wilhelm Weitling, but they did not use the word anarchist or anarchism in describing themselves or their beliefs.
  The first political philosopher to call himself an anarchist was Pierre-Joseph Proudhon, marking the formal birth of anarchism in the mid-19th century.
  Since the 1890s and beginning in France, the term libertarianism has often been used as a synonym for anarchism and its use as a synonym is still common outside the United States.
  On the other hand, some use libertarianism to refer to individualistic free-market philosophy only, referring to free-market anarchism as libertarian anarchism.
  While opposition to the state is central, defining anarchism is not an easy task as there is a lot of talk among scholars and anarchists on the matter and various currents perceive anarchism slightly differently.
  Inserting some random text to make sure it is extend qwertyuiop asdfghjkl zxcvbnm
  " >> test_data_folder/raw_txt/test.txt
done