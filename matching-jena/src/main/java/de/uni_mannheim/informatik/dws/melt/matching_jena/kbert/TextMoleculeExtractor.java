package de.uni_mannheim.informatik.dws.melt.matching_jena.kbert;

import de.uni_mannheim.informatik.dws.melt.matching_jena.TextExtractor;

import java.util.stream.Stream;

public interface TextMoleculeExtractor extends TextExtractor {

    Stream<String> getIndexStream();
    void emptyCache();
}
