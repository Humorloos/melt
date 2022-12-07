package de.uni_mannheim.informatik.dws.melt.matching_jena.kbert;

import de.uni_mannheim.informatik.dws.melt.matching_jena.TextExtractor;

import java.util.stream.Stream;

/**
 * An interface which extracts resources of a given OntModel.
 * This can be for example all classes, all properties, all object properties etc.
 */
public interface TextMoleculeExtractor extends TextExtractor {

    Stream<String> getIndexStream();
    void emptyCache();
}
