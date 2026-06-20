# NLPCC Migration Notes

## Source

Initial source material came from local ignored path:

`G:\Research\BotDetection\paper\NLPCC`

## Migrated

- LaTeX manuscript entrypoint and section source.
- Bibliography and LNCS style/class files required by the source.
- Writing guidelines, paper plan, evidence matrix, narrative notes, and
  improvement log.
- Editable and regenerable figure/table sources.

## Not Migrated

- Compiled PDFs.
- TeX build products.
- Preview and rendered-page images.
- Portable TeX runtime directories.
- Python bytecode.
- Duplicate revision snapshots and one-off probe files.

## Rationale

The tracked `NLPCC/` tree is intended to be reviewable and reproducible without
turning the parent repository into a build cache or artifact store.
