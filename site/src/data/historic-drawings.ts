export interface HistoricDrawing {
  artist: string;
  year: string;
  title: string;
  image: string;
  alt: string;
  source: string;
}

export const historicDrawings: HistoricDrawing[] = [
  {
    artist: 'Leonardo da Vinci',
    year: 'c. 1490',
    title: 'Homem Vitruviano',
    image: '/images/archive/vitruvian.jpg',
    alt: 'Estudo de proporções humanas de Leonardo da Vinci, desenhado a tinta.',
    source: 'https://commons.wikimedia.org/wiki/File:Vitruvian.jpg',
  },
  {
    artist: 'Charles Darwin',
    year: '1837',
    title: '“I think”',
    image: '/images/archive/darwin-tree.jpg',
    alt: 'Página do caderno de Darwin com a árvore da vida e as palavras “I think”.',
    source: 'https://commons.wikimedia.org/wiki/File:Darwin_Tree_1837.png',
  },
  {
    artist: 'Boulton & Watt',
    year: '1784',
    title: 'Motor de Watt',
    image: '/images/archive/watt-steam-patent.jpg',
    alt: 'Esboço técnico do motor a vapor de Boulton & Watt com peças e mecanismos indicados.',
    source: 'https://commons.wikimedia.org/wiki/File:SteamEngine_Boulton%26Watt_1784.png',
  },
  {
    artist: 'Orville e Wilbur Wright',
    year: '1903 · pedido',
    title: 'Flying machine',
    image: '/images/archive/wright-flying-machine.jpg',
    alt: 'Desenho técnico da máquina voadora dos irmãos Wright, patente requerida em 1903.',
    source: 'https://patents.google.com/patent/US821393A/en',
  },
  {
    artist: 'Alexander Graham Bell',
    year: '1876',
    title: 'Patente do telefone',
    image: '/images/archive/bell-telephone.jpg',
    alt: 'Folha de desenhos da patente de Bell, com diagramas dos aparelhos e do circuito.',
    source: 'https://patents.google.com/patent/US174465A/en',
  },
  {
    artist: 'Robert Metcalfe',
    year: '1976',
    title: 'Diagrama da Ethernet',
    image: '/images/archive/ethernet-1976.gif',
    alt: 'Diagrama desenhado à mão de uma estação ligada ao cabo Ether.',
    source: 'https://grouper.ieee.org/groups/802/3/ethernet_diag.html',
  },
  {
    artist: 'Tim Berners-Lee',
    year: '1989',
    title: 'Proposta para a Web',
    image: '/images/archive/www-proposal.gif',
    alt: 'Página da proposta de gestão de informação de Tim Berners-Lee, com seu diagrama de conexões.',
    source: 'https://info.cern.ch/Proposal.html',
  },
  {
    artist: 'Apple',
    year: '2007 · pedido',
    title: 'Design do iPhone',
    image: '/images/archive/iphone-patent.jpg',
    alt: 'Desenho de patente do iPhone original, vista perspectiva do aparelho.',
    source: 'https://patents.google.com/patent/USD558756S1/en',
  },
];
