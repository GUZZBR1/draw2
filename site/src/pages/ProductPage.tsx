import { Link } from 'react-router-dom';

export default function ProductPage() {
  return (
    <main className="product-page" id="page-content">
      <section className="product-content">
        <Link className="product-status group" to="/vision" aria-label="Em construção. Leia a visão do Draw.">
          <span className="status-dot" />
          em construção
          <span className="status-arrow" aria-hidden="true">↗</span>
        </Link>
        <figure className="construction-art">
          <img
            src="/images/product-construction-diorama.png"
            alt="Um macaco de capacete conversa com um australopiteco sobre uma planta de obra, enquanto um neandertal segura uma britadeira ao lado de uma maquete."
            draggable={false}
          />
        </figure>
        <h1 className="product-headline">
          Do primeiro traço
          <br />
          <em>à próxima geração.</em>
        </h1>
        <p className="product-description">
          Estamos construindo uma ponte entre o papel e as ferramentas que projetam e testam o
          mundo. A ideia começa à mão; o que ela pode se tornar ainda está sendo desenhado.
        </p>

        <Link className="vision-cta" to="/vision">
          Leia a visão <span aria-hidden="true">↗</span>
        </Link>
      </section>
    </main>
  );
}
