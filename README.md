# Effect of Key Match Events on Football Passmaps
## Authors:
- Vuk Ðuranović
- Maruša Oražem
- Vid Stropnik

## About
Welcome to the repository for the course project 'Effect of Key Match Events on Football Passmaps', as part of the Introduction to Network Analysis course at the Faculty of Computer and Information Science, University of Ljubljana (2020/21)

The opening goal, player dismissal and the half-time break are justsome of the several events in any given football match, that mightincur a fundamental change in a team’s approach towards the game.If the team’s behaviour before and after such an event is to be mod-elled as a pair of directed interaction networks with weighted edgesbetween players, interesting changes in inferred graphs might be ob-served.  Our team proposes to work on this problem as part of ourIntroduction  to  Network  Analysis  course  project.   In  it,  we  plan  toobserve the changes in graph centrality measures, community struc-tures, motif significance profiles and graphlet agreement for such asplit,  introducing a novel method of football match analysis usingnetwork science.

The present repository shows our progress in solving the described problems, with the goal of finishing in mid-june 2021.

## Repository
Please see `docs` for available reports and documentation to better familiarize yourself with the data.

In `nets`, the exported networks are available and ready for processing.

`src` shows the code so far. Currently, it includes the `event.json` parser for the data available at [https://github.com/statsbomb/open-data](Statsbomb's Open Data repository).
