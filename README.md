# Description
Code for paper "WakaVT: A Sequential Variational Transformer for Waka Generation"<br>

WakaVT is a deep Waka generation model proposed for automatically producing Waka poems with user-specified keywords as input. In consideration of poems' overall quality including novelty, diversity and linguistic quality, we developed WakaVT by combining the latent variable model and Transformer architecture. Besides, we proposed the Fused Multilevel Self Attention Mechanism (FMSA) to model the hierarchical linguistic structure of Waka, which further improved the linguistic quality of the generated poems. Experimental results demonstrated that WakaVT outperformed multiple baselines significantly.

The following table shows some examples generated by WakaVT+FMSA model. The keywords used to generate each poem is highlighted in bold. We provided the English translation of each poem to help those not expert in ancient Japanese understand the meaning of the generated poems. We also invited 3 experts to give comments on each poem individually. Their comments confirmed that WakaVT+FMSA could indeed generate ingenious and innovative Waka poems.

<table>
  <thead>
    <tr >
        <td  align="center">ID</td>
        <td  colspan="2"  align="center">Sample</td>
    </tr>
  </thead>
  <tbody>
    <tr >
        <td  rowspan="3"  align="center">1</td>
        <td  align="center">Poem</td>
        <td  align="center">あけてゆく−みねのこのはの−こずゑより−はるかにつづく−さをしかの<b>こゑ</b></td>
    </tr>
    <tr >
        <td  align="center">Translation</td>
        <td  align="center">The morning light climbs the treetops of the top mountain, and the <b>cry</b> of stags spreads far and wide.</td>
    </tr>
    <tr >
        <td  align="center">Comments</td>
        <td>
          <ol>
            <li>The scenery is <i>fresh with a wild imagination</i>, and the connection of several images is ingenious.</li>
            <li>This poem is <i>thought-provoking</i> as it gradually transitions from visual to auditory. The far-reaching cry of the stag can further arouse the longing for the loved one. It is indeed a classical Waka poem.</li>
            <li>This Waka is <i>beautiful as well as touching</i> under the background of autumn. The author expresses parting sadness through the auditory perspective of the stag's courtship sound.</li>
          </ol>
        </td>
    </tr>
    <tr >
        <td  rowspan="3"  align="center">2</td>
        <td  align="center">Poem</td>
        <td  align="center">こひしさは−ひとのこころに−さよふけて−わが<b>なみだ</b>こそ−おもひしらるれ</td>
    </tr>
    <tr >
        <td  align="center">Translation</td>
        <td  align="center">Night approaches, my love is longing in his heart faraway, my <b>tears</b> come and see by all.</td>
    </tr>
    <tr >
        <td  align="center">Comments</td>
        <td>
          <ol>
            <li>The expression is <i>proficient and sophisticated with strong feeling</i>. There is a jump in content between the first two phrases and the last three phrases.</li>
            <li>This poem is <i>grammatically coherent and accurate</i>, which reminds us of the primal「恋しさに思ひみだれてねぬる夜の深き夢ぢをうつうともがな」, a sleepless night troubled by love.</li>
            <li>This poem is <i>simple in language and sincere in emotion</i>. The poet depicts the pain of lovesickness and reveals his infinite sadness for incomplete love.</li>
          </ol>
        </td>
    </tr>
    <tr >
        <td  rowspan="3"  align="center">3</td>
        <td  align="center">Poem</td>
        <td  align="center">ふるさとの−あとをたづねて−なつくさの−しげみにかかる−をのの<b>かよひぢ</b></td>
    </tr>
    <tr >
        <td  align="center">Translation</td>
        <td  align="center">I trace the road of my hometown, the lush grass of summer covers the return <b>path</b> in the wilderness.</td>
    </tr>
    <tr >
        <td  align="center">Comments</td>
        <td>
          <ol>
            <li>The poet’s longing for home is reflected. It is quite <i>interesting</i> to know how the fourth phrase connects the third and fifth phrases.</li>
            <li>Lush summer grass covered the trace of the hometown road. The entire poem is <i>closely connected and understandable in meaning</i>. The accurate grasp of the image of summer grass is quite touching.</li>
            <li>This poem is composed with plain language, yet it conveys authentic emotion. It is <i>simple in style, beautiful in artistic conception, and fluent in language</i>. The poet truly expresses profound nostalgia.</li>
          </ol>
        </td>
    </tr>
    <tr >
        <td  rowspan="3"  align="center">4</td>
        <td  align="center">Poem</td>
        <td  align="center">はてもなき−まがきのくさは−おく<b>つゆ</b>に−おもひあまりて−やどるつきかな</td>
    </tr>
    <tr >
        <td  align="center">Translation</td>
        <td  align="center">Endless weeds by the fence are covered with <b>dew</b>, yearning was reflected on the dew by moonlight.</td>
    </tr>
    <tr >
        <td  align="center">Comments</td>
        <td>
          <ol>
            <li>The idea of connecting certain images such as weeds, dew, and moon is quite <i>intriguing</i>. The fourth phrase is <i>affectionate</i>.</li>
            <li>The weeds beside the fence correspond to melancholy, and dew corresponds to the moon. This poem is <i>quite classical and interesting</i>, and it feels like <i>Kokinshu</i>.</li>
            <li>This Waka is <i>beautiful in artistic conception</i>. The dew and the moon's mutual reflection presents an ethereal and tranquil scene, revealing the poet’s romantic feelings of nature.</li>
          </ol>
        </td>
    </tr>
  </tbody>
 </table>

# Documentation
Data for training and testing can be found at https://db.nichibun.ac.jp/pc1/ja/category/waka.html<br>
Full documentation will be added later.
