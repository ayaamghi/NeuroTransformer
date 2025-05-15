import { BaseInputParams, InputBindingPlugin } from '@tweakpane/core';
export interface TextareaPluginInputParams extends BaseInputParams {
    view: 'textarea';
    rows?: number;
    placeholder?: string;
}
export declare const TweakpaneTextareaPlugin: InputBindingPlugin<string, string, TextareaPluginInputParams>;
