export default function VisionPage() {
  return (
    <main className="vision-page" id="page-content">
      <header className="vision-intro">
        <p className="page-eyebrow">uma visão para o desenho</p>
        <h1 className="vision-headline">
          O papel é só
          <br />
          <em>o início.</em>
        </h1>
      </header>

      <div className="vision-copy">
        <p>
          A gente pensa primeiro no papel. Antes de abrir menus, escolher ferramentas ou
          ajustar camadas, vem um gesto simples: desenhar para descobrir.
        </p>
        <p>
          Draw transforma esse gesto em uma ponte em tempo real. Uma câmera acompanha os
          traços; visão computacional os converte em modelos que podem chegar ao CAD, à
          robótica e a simuladores — sem interromper o pensamento.
        </p>
        <p>
          A visão é que o desenho não termine na captura. Uma linha pode se tornar uma peça,
          uma parede ou o caminho de um robô. A simulação devolve respostas enquanto a pessoa
          ainda desenha, para que ideia e modelo evoluam juntos.
        </p>
        <p>
          Começamos pela fundação: capturar cada traço com fidelidade, inclusive ao apagar e
          corrigir. O destino é mais ambicioso: pensar no papel e criar diretamente nos
          sistemas que projetam, testam e dão forma ao mundo.
        </p>
      </div>

      <p className="vision-signoff">do papel à vida, sem perder o gesto.</p>
    </main>
  );
}
