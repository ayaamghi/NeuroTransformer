import { Value, View, ViewProps } from '@tweakpane/core';
interface Config {
    value: Value<string>;
    viewProps: ViewProps;
    rows: number;
    placeholder: string;
}
export declare class TextAreaView implements View {
    readonly inputElement: HTMLTextAreaElement;
    readonly element: HTMLElement;
    private readonly value_;
    constructor(doc: Document, config: Config);
    refresh(): void;
    private onChange_;
}
export {};
