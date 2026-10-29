# Material reference: printed PETG and PLA, from data sheets

This file is a read-only reference. Topopt does not read it. Topopt has no default allowable stress (decision D7, revised on 2026-10-24: a sourced filament catalog
ships and gives the modulus and the density of a filament, `find_material`; it gives no allowable stress).
The allowable is your choice. This file shows what makers publish, so that you can choose `allowable_mpa`, write `allowable_source`
and choose a `safety_factor` for a load case. `topopt verify` takes the same choice as `--allowable-mpa`.

All data comes from data sheets that were opened on 2026-10-22. The address of each sheet is in the section "Sheets". A number
that has no sheet is not in this file. Other materials and other makers are not in this file.

## How to read the numbers

- Each sheet reports tensile tests of bars that the maker printed. The standard is ISO 527. Some sheets also name GB/T 1040 or ASTM D638.
- The values are "typical values" of one maker, one printer and one set of print settings. They describe a bar, not your part.
- "X-Y" means that the load runs in the layer plane. "Z" means that the load runs across the layers. In every sheet with a Z value, the Z value is lower than the X-Y value.
- The sheets do not use the same words:
  - Polymaker and Bambu Lab give "X-Y" and "Z". They write "tensile strength" and do not say if it is the yield stress or the peak stress.
  - Prusament gives "horizontal" and "vertical xz". Its drawing shows the "vertical xz" bar on its edge. Its load still runs in the layer plane.
    So "vertical xz" is not a value across the layers. Prusament gives the "tensile yield strength".
  - AzureFilm PLA Prime gives XY, XZ and ZX. Its drawing shows the ZX bar standing upright, so ZX is across the layers.
  - AzureFilm PETG gives one "tensile yield stress" and does not say how the bar was printed.
- The sign "±" is as printed in each sheet. The sheets do not say what it is, or how many bars were tested.

## PETG: tensile strength of printed bars

Numbers in square brackets are the sheets in the section "Sheets". Units: MPa unless the cell says GPa.

| Sheet | In the layer plane | Across the layers (Z) | Named in the sheet | Modulus: in plane / Z |
|---|---|---|---|---|
| Polymaker PolyLite PETG, version 5.4 [1] | 50.8 ± 0.9 | 42.8 ± 2.8 | tensile strength | 2116.8 ± 68.1 / 1898.7 ± 98.5 |
| Polymaker PETG (a newer product), version 3.0 [2] | 47.96 ± 4.88 | 45.71 ± 1.76 | tensile strength | 2311.11 ± 92.41 / 2202.91 ± 52.34 |
| Prusament PETG, version 1.1, 16-02-2022 [3] | 47 ± 2 (horizontal); 50 ± 1 (vertical xz) | no tensile value; "interlayer adhesion" 18 ± 4 (the maker's own method) | tensile yield strength | 1.5 ± 0.1 GPa (horizontal); 1.6 ± 0.1 GPa (vertical xz) |
| AzureFilm PETG, no version or date on the sheet [4] | 51 (direction not stated) | not given | tensile yield stress; stress at break 20 | 2980 (direction not stated) |
| Bambu Lab PETG Basic, version 2.0 [5] | 48 ± 5 | 39 ± 3 | tensile strength | 1460 ± 190 / 1120 ± 130 |

## PLA: tensile strength of printed bars

| Sheet | In the layer plane | Across the layers (Z) | Named in the sheet | Modulus: in plane / Z |
|---|---|---|---|---|
| Polymaker PolyLite PLA, version 5.4 [8] | 52.3 ± 0.3 | 40.5 ± 0.5 | tensile strength | 3426.9 ± 64.8 / 3064.9 ± 83.4 |
| Prusament PLA, version 1.1, 27-07-2022 [9] | 51 ± 3 (horizontal); 59 ± 2 (vertical xz) | no tensile value; "interlayer adhesion" 17 ± 3 (the maker's own method) | tensile yield strength | 2.3 ± 0.1 GPa (horizontal); 2.4 ± 0.1 GPa (vertical xz) |
| AzureFilm PLA Prime (an "engineering grade" PLA), TDS 04.24 [10] | XY 27.8; XZ 31.4 | ZX 14.2 | tensile strength; yield strength XY 26.1, XZ 29.7, ZX 14.5 | XY 2400, XZ 2600, ZX 1900 |
| Bambu Lab PLA Basic, version 2.0, dry state [11] | 39 ± 2 | 35 ± 3 | tensile strength | 2680 ± 130 / 2160 ± 90 |

## How the bars were made

The sheets give these conditions. A condition that a sheet does not give is marked "not stated". The numbers in brackets are the sheets.

| Sheet | Printer and slicer | Layer height | Infill and walls | Nozzle / bed temperature; print speed | Other |
|---|---|---|---|---|---|
| Polymaker PolyLite PETG [1] | not stated | not stated | 100 % infill, 2 shells, 3 top and bottom layers | 240 °C / 80 °C; not stated | fan off; room at ambient temperature |
| Polymaker PETG [2] | not stated | not stated | 100 % infill, 2 shells, 3 top and bottom layers | 240 °C / 80 °C; not stated | fan off; room at ambient temperature |
| Prusament PETG [3] | Original Prusa i3 MK3, Slic3r Prusa Edition 1.40.0 | 0.20 mm | 100 % rectilinear infill, 2 perimeters, 0 top and 0 bottom solid layers | 250 °C / 80 °C; 200 mm/s | "horizontal" bar lies flat; "vertical xz" bar stands on its edge |
| AzureFilm PETG [4] | AzureFilm printer, Cura, 0.4 mm nozzle | 0.2 mm | 100 % infill; shells not stated | 230 °C / 80 to 90 °C; 50 mm/s | |
| Bambu Lab PETG Basic [5] | not stated | not stated | 100 % infill | 255 °C / 70 °C; 150 mm/s | "All the specimens were annealed and dried at 65 °C for 8 hours before testing" |
| Polymaker PolyLite PLA [8] | not stated | not stated | 100 % infill, 2 shells, 3 top and bottom layers | 230 °C / 50 °C; not stated | fan on; room at ambient temperature |
| Prusament PLA [9] | Original Prusa i3 MK3, Slic3r Prusa Edition 1.40.0 | 0.20 mm | 100 % rectilinear infill, 2 perimeters, 0 top and 0 bottom solid layers | 215 °C / 60 °C; 200 mm/s | |
| AzureFilm PLA Prime [10] | not stated | 0.2 mm | 100 % infill, 2 shells | not stated for the bars (the sheet recommends 230 °C / 60 °C) | |
| Bambu Lab PLA Basic [11] | not stated | not stated | 100 % infill | 220 °C / 35 °C; 200 mm/s | "All the specimens were annealed and dried at 55 °C for 8 hours before testing" |

No sheet in this list states the speed or the temperature of the tension test. The Polymaker and Bambu Lab drawings show a bar 150 mm long
and 4 mm thick. It is 10 mm wide in the narrow part and 20 mm wide at the ends. The other sheets give no size.

## Ranges in these sheets

These are the lowest and the highest central values of the tables above. They are not design values. Each range mixes makers, products,
print settings and sheet versions.

| Material | In the layer plane | Across the layers (Z) | Z over in-plane, in the same sheet |
|---|---|---|---|
| PETG | 47 to 51 MPa (five sheets; the direction of one is not stated) | 39 to 45.71 MPa (three sheets) | 0.81 to 0.95 (three sheets) |
| PLA | 27.8 to 52.3 MPa for the flat bars (four sheets); the bars on the edge give 31.4 and 59 | 14.2 to 40.5 MPa (three sheets) | 0.51 to 0.90 (three sheets) |

Prusament gives no tensile value across the layers. Its "interlayer adhesion" is 18 ± 4 MPa for PETG [3] and 17 ± 3 MPa for PLA [9], by the
maker's own method. The sheet does not describe that method.

## What the numbers leave out

- **Version of the sheet.** One product can have very different numbers in two versions of its sheet. PolyLite PETG, tensile strength X-Y:
  31.9 ± 1.1 MPa in version 3.0 (Jun. 2017) [6] and in version 4.0 (Nov. 2018) [7], and 50.8 ± 0.9 MPa in version 5.4 [1]. The old
  versions name ASTM D638, give no Z value and state a printing speed of 45 mm/s and conditioning for 24 h at room temperature. The sheets do not
  explain the change. A number from an old sheet, or from a shop page that copies one, can be far from the current sheet. Check the version.
- **Time under load.** No sheet in this list gives creep, stress relaxation or fatigue data. A tension test loads a bar once, up to
  failure. A plastic part under a steady load can keep deforming, and can fail at a stress below the stress of such a test.
- **Temperature.** No sheet gives the strength at another temperature. The heat deflection temperatures (ISO 75, a bending test under 0.45 MPa or 1.8 MPa)
  are 65 to 78 °C for PETG [1][2][3][5] and 54 to 60 °C for PLA [8][9][11]. The AzureFilm PLA Prime sheet gives 95 °C with an asterisk and no note [10].
  The sheets give nothing for a warm room or for a wet part.
- **Bars are not parts.** All bars have 100 % infill and one bar shape. Your part has other infill, walls, layer height, nozzle, printer,
  corners and holes. Two sheets say that the bars were dried and annealed before the test [5][11]. The other sheets do not say how the bars were stored.
- **Direction.** Topopt checks the von Mises stress of an isotropic material. It does not know the print direction. A value across the
  layers is a tensile strength across the layers. It is not a von Mises limit. The choice of the value for your part is yours.
- **Scatter.** The "±" shows the scatter of one set of bars of one maker. It does not cover other printers, other spools or other makers.
- **The allowable is your choice.** A tensile strength from a sheet is not an allowable stress. The allowable depends on the material, the print
  direction, the time under load and the safety factor that you choose. No sheet gives a safety factor, and this file gives none.
- **The makers' own words.** The Polymaker sheets say the values "should not be used for design specifications or quality control purposes" [1][2][6][7][8].
  Prusament says "just for your information and comparison" [3][9]. Bambu Lab says "for design reference and comparison only" [5][11].
  AzureFilm says the data "should not be used to establish specification limits or used alone as the basis of design" [4].

## How to use it in a YAML or in `topopt verify`

1. Choose the value that fits your part and its print direction. A range above is a fact about the sheets, not a recommendation.
2. Choose a safety factor for your case. This file gives none.
3. For the optimizer, write the choice in the load case. `allowable_mpa` adds a stress limit row: the limit is `allowable_mpa / safety_factor`.
   The source is free text:

```yaml
    allowable_mpa: <MPa>
    safety_factor: <factor>
    allowable_source: "<maker> <product> data sheet <version>, <property>, <value> MPa, printed ISO 527 bars"
```

4. For `topopt verify`, give the limit itself: `--allowable-mpa <MPa>` (the tool `verify_run` has `allowable_mpa`). Verify applies no safety factor.
   It compares the stress of each load case, at the percentile by volume, with that number. Without it, the result has no limit block and no
   pass or fail.

## Sheets

All sheets were opened on 2026-10-22. The address is the one that was opened. A sheet that another company hosts carries the maker's name.
It was not compared with the maker's own file.

1. Polymaker PolyLite PETG, version 5.4. Wiki page (last updated "2 months ago" on that day):
   https://wiki.polymaker.com/polymaker-products/more-about-our-products/documents/technical-data-sheets/petg-pet/polylite-tm-petg
   and its PDF: https://cdn.shopify.com/s/files/1/0548/7299/7945/files/PolyLite_PETG_TDS_EN_V5.4.pdf?v=1731954068 (same numbers).
2. Polymaker PETG (a newer product), version 3.0, PDF linked from
   https://wiki.polymaker.com/polymaker-products/more-about-our-products/documents/technical-data-sheets/petg-pet/polymaker-tm-petg-new :
   https://cdn.shopify.com/s/files/1/0548/7299/7945/files/PETG_TDS_Polymaker_297e875c-5d38-4d2b-ba68-eef8916de183.pdf?v=1772830046
3. Prusament PETG, version 1.1, last update 16-02-2022 (a copy hosted by a reseller):
   https://www.3dprintergear.com.au/assets/files/technical-data-sheet%20(1).pdf
4. AzureFilm PETG TDS: https://azurefilm.com/wp-content/uploads/2023/10/PETG_TDS1-2.pdf
5. Bambu Lab PETG Basic, technical data sheet version 2.0 (a copy hosted by the reseller Additive-X):
   https://www.additive-x.com/shop/media/mageplaza/product_attachments/attachment_file/b/a/bambu_petg_basic_technical_data_sheet_12.pdf
   (reached through https://www.additive-x.com/shop/mpattachments/file/viewonline/id/580/product_id/2330)
6. Polymaker PolyLite PETG, version 3.0, Jun. 2017 (a copy hosted by the printer maker LulzBot):
   https://devel.lulzbot.com/filament/polymaker/PETG/SDS/PolyLite%e2%84%a2%20PETG_TDS_V3.pdf
7. Polymaker PolyLite PETG, version 4.0, Nov. 2018 (a copy hosted by a reseller, short link that redirects to a PDF):
   https://matterhackers.com/r/T6szyy
8. Polymaker PolyLite PLA, version 5.4. Wiki page:
   https://wiki.polymaker.com/polymaker-products/more-about-our-products/documents/technical-data-sheets/pla/polylite-tm-pla
   and its PDF: https://cdn.shopify.com/s/files/1/0548/7299/7945/files/PolyLite_PLA_TDS_EN_V5.4.pdf?v=1731949179 (same numbers).
9. Prusament PLA, version 1.1, last update 27-07-2022 (a copy hosted by a reseller, short link that redirects to a PDF):
   https://www.matterhackers.com/r/xFvGl8
10. AzureFilm PLA Prime, technical data sheet 04.24: https://azurefilm.com/wp-content/uploads/2024/04/TDS-PLA-Prime.pptx.pdf
11. Bambu Lab PLA Basic, technical data sheet version 2.0 (a copy hosted by the reseller Additive-X):
    https://www.additive-x.com/shop/media/mageplaza/product_attachments/attachment_file/b/a/bambu_pla_basic_technical_data_sheet_ax_1.pdf
    (reached through https://www.additive-x.com/shop/mpattachments/file/viewonline/id/494/product_id/2236)

The Z values come from the sheets 1, 2, 5, 8, 10 and 11. The sheets 3, 4 and 9 give no tensile value across the layers.
