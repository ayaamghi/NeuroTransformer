import { Controller, Value, ViewProps } from '@tweakpane/core';
import { TextAreaView } from './view.js';
/**
 * @hidden
 */
export interface Config {
    value: Value<string>;
    viewProps: ViewProps;
    rows: number;
    placeholder: string;
}
/**
 * @hidden
 */
export declare class TextAreaController implements Controller<TextAreaView> {
    readonly value: Value<string>;
    readonly view: TextAreaView;
    readonly viewProps: ViewProps;
    readonly rows: number;
    readonly placeholder: string;
    constructor(doc: Document, config: Config);
    private onInputChange_;
}
