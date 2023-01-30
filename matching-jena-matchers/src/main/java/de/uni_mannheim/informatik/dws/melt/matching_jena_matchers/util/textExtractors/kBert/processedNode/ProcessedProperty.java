package de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.processedNode;

import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.PropertyVocabulary;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.constant.KBertLabelPropertyTypes;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.vocabulary.RDFS;

import static de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.constant.KBertLabelPropertyTypes.*;

/**
 * Property wrapper for text extraction
 */
public class ProcessedProperty extends ProcessedResource<Property> {
    public ProcessedProperty(Property property) {
        super(property);
    }

    public KBertLabelPropertyTypes getLabelType() {
        return resource.equals(RDFS.label) ? LABEL
                : PropertyVocabulary.LABEL_LIKE_PROPERTIES.contains(resource) ? LABEL_LIKE
                : PropertyVocabulary.hasPropertyLabelFragment(resource) ? LABEL_NAME
                : OTHER;
    }
}
